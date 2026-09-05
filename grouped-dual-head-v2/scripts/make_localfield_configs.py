#!/usr/bin/env python3
"""Build local_field full-dataset configs by transforming the known-good v63 config.

Starting from a complete, working config guarantees every required key is present;
we only strip the dead adapter/expert/recovery machinery and repoint the run at the
W2 warm-start parent + width-128 base + a zero-init local_field residual.

Produces two configs:
  * gate3_short : a few epochs, small validation panel — the independent full-data
                  short test (verify descent, no leak/OOM/NaN).
  * gate4_long  : long schedule for the 4-GPU survival run.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs/saved_time_v4/generated/v63_full_dataset_allband_pilot_4gpu.yaml"
ART = "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_residual"
W2_CKPT = (
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "capacity_ladder/W2_w128_d12/checkpoints/update_0800.pt"
)
PARENT_IDENTITY = f"{ART}/w2_parent_identity.json"
BASE_W128 = "configs/grouped_v3/continuous_pilot_w128.yaml"


def base_config() -> dict:
    cfg = yaml.safe_load(SRC.read_text())

    # --- strip the dead additive-head machinery entirely ---
    for dead in ("family_experts", "band_limited_adapter", "residual_recovery", "family_curriculum"):
        cfg.pop(dead, None)
    cfg["variant_overrides"] = {"local_field": True, "local_field_residual": True}

    # --- repoint at the W2 warm-start parent + width-128 base ---
    cfg["base_config"] = BASE_W128
    cfg["parent_checkpoint"] = W2_CKPT
    cfg["parent_identity"] = PARENT_IDENTITY
    cfg.pop("parent_checkpoint_identity", None)  # defaults to parent_identity
    cfg["checkpoint_transfer"] = {
        "allow_parent_manifest_mismatch": False,
        "parent_optimizer_state": False,
        "allow_new_local_field_parameters": True,
    }

    # --- loss: the validated coarse+correction recovery objective (delta=0.5) ---
    cfg["loss"] = {
        "delta": 0.5,
        "delta_energy_floor_fraction": 0.1,
        "hard_causality": True,
        "hard_causality_lead_cycles": 1.0,
        "spatial_gradient": 0.1,
        "spectrum": 0.0,
        "temporal_difference": 0.0,
    }

    # --- optimizer: dense group carries local_field (see DENSE_PREFIXES); no
    #     temporal/expert/band-adapter groups. Give the fresh U-Net clip headroom. ---
    opt = cfg["optimizer"]
    for dead in (
        "temporal_basis_learning_rate",
        "family_expert_learning_rate",
        "band_adapter_feature_learning_rate",
        "band_adapter_output_learning_rate",
        "training_loss_objective",
        "training_loss_time_block",
    ):
        opt.pop(dead, None)
    opt["dense_learning_rate"] = 1.0e-4
    opt["geometry_learning_rate"] = 2.0e-5
    opt["backbone_learning_rate"] = 2.0e-6
    opt["weight_decay"] = 1.0e-6
    opt["full_forward_checkpointing"] = True
    opt["training_time_block"] = 1
    opt["adamw_implementation"] = "fused"
    opt["gradient_clip"] = 1.0
    opt["gradient_clip_mode"] = "prefix_limits"
    opt["gradient_clip_prefix_limits"] = {
        "coordinate_encoder": 5.0,
        "default": 1.0,
        "dense_decoder": 20.0,
        "local_field": 80.0,
        "fusion": 5.0,
        "medium_encoder": 2.0,
        "source_encoder": 5.0,
        "travel_branch": 5.0,
    }

    # --- memory: width-128 residual runs both MIONet coarse and the U-Net; the
    #     capacity-ladder probe peaked at ~20 GiB with one record + 16 frames, so
    #     force a single-record physical microbatch and accumulate for batch. ---
    cfg["microbatch_records"] = 1
    cfg["macro_records"] = 8
    cfg["macros_per_update"] = 4  # 4-GPU: one macro per rank
    cfg["training_frames_per_record"] = 16
    cfg["time_appearance_offset"] = 15
    cfg["time_policy"] = "appearance16"
    cfg["query_points"] = 1
    cfg["workers"] = 8
    cfg["prefetch_factor"] = 4
    cfg["seed"] = 372
    cfg["energy_floor_fraction"] = 0.01
    cfg["travel_time_h5"] = "/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5"

    cfg["gate"] = {
        "external_evidence_gate": False,
        "family_regression_tolerance": 0.03,
        "maximum_peak_cuda_gib": 23.5,
        "pilot_epochs": 2,
        "target_aggregate_relative_l2": 0.1,
        "target_family_relative_l2": 0.12,
    }
    return cfg


def main() -> int:
    out_dir = ROOT / "configs/saved_time_v4/generated"

    short = base_config()
    short["artifact_dir"] = f"{ART}/gate3_short"
    short["epochs"] = 3
    short["optimizer"]["schedule_total_epochs"] = 3
    short["optimizer"]["schedule_epoch_offset"] = 0
    short["optimizer"]["warmup_epochs"] = 1
    short["validation"] = {
        "all_records_every": 999,     # skip the expensive all-records pass in the short test
        "final_frames_per_record": 32,
        "frames_per_record": 24,
        "full_panel_records": 12,
        "microbatch_records": 1,
        "panel_records": 12,
    }
    (out_dir / "local_field_w128_residual_gate3_short_4gpu.yaml").write_text(
        yaml.safe_dump(short, sort_keys=True)
    )
    print("WROTE gate3_short")

    long = base_config()
    long["artifact_dir"] = f"{ART}/gate4_long"
    long["epochs"] = 40
    long["optimizer"]["schedule_total_epochs"] = 40
    long["optimizer"]["schedule_epoch_offset"] = 0
    long["optimizer"]["warmup_epochs"] = 2
    long["validation"] = {
        "all_records_every": 5,
        "final_frames_per_record": 401,
        "frames_per_record": 32,
        "full_panel_records": 48,
        "microbatch_records": 1,
        "panel_records": 48,
    }
    (out_dir / "local_field_w128_residual_gate4_long_4gpu.yaml").write_text(
        yaml.safe_dump(long, sort_keys=True)
    )
    print("WROTE gate4_long")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
