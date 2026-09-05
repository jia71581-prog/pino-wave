#!/usr/bin/env python3
"""Generate the preregistered AIS-MQFNO experiment configurations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


MIXTURE_NAMES = ("uniform", "interface", "source_wavefront", "residual", "edge", "receiver")
LOSS_NAMES = ("receiver", "phase", "local_spectrum", "energy")

COMMON: dict[str, Any] = {
    "seed": 20260714,
    "data": {
        "path": "/home/jiayh/Data/data/pino.hdf5",
        "split_manifest": "artifacts/hybrid_drp_pino/fno_continue_pretrain_optimizer_safe/splits.json",
        "tensor_axes": ["sample", "time", "x", "z"],
    },
    "normalization": {"stats_path": "artifacts/pino_adaptation/normalization_stats.json"},
    "sampling": {"max_time_steps": 160},
    "receiver": {
        "z_m": [20.0], "x_start_m": 0.0, "x_stop_m": 1960.0, "x_stride_m": 40.0,
    },
    "model": {
        "name": "ais_mqfno", "global_in_features": 6, "native_in_channels": 5,
        "global_max_size": 100, "spatial_width": 24, "spatial_modes": 24,
        "spatial_layers": 3, "temporal_modes": 32, "local_dim": 16,
        "fusion_dim": 32, "local_patch_size": 17, "activation_checkpointing": True,
    },
    "train": {
        "batch_size": 1, "query_sites_per_scene": 2048,
        "receiver_sites_per_scene": 50, "patch_centers_per_scene": 4,
        "patch_size": 17, "query_chunk_size": 256, "optimizer": "adamw",
        "weight_decay": 1e-4, "grad_clip": 1.0, "amp": False,
        "scheduler": "cosine_by_update",
    },
    "loss": {
        "hh_reweight": True, "late_time_weights": [1.0, 1.0, 1.0, 1.25],
        "receiver_weight": 0.0, "phase_weight": 0.0,
        "local_spectrum_weight": 0.0, "energy_weight": 0.0,
        "pde_weight": 0.0, "drp_weight": 0.0,
    },
    "sampler": {
        "tile_grid": [16, 16], "ema_momentum": 0.9, "lag_epochs": 1,
        "residual_power": 1.0, "min_probability": 1e-12,
    },
    "experiment": {
        "gpu_hour_budget": 72.0, "query_unit": "spatial_site_full_trace",
        "optimizer_updates": 12000, "physical_scene_draws": 12000,
        "diagnostic_only": False,
    },
}

SAMPLER_ARMS = {
    "b1_uniform": ([1.0, 0, 0, 0, 0, 0], True, False),
    "s_static_hh": ([0.375, 0.25, 0.1875, 0, 0.0625, 0.125], True, False),
    "a_residual_uncorrected": ([0.30, 0, 0, 0.70, 0, 0], False, True),
    "ai_adaptive_hh": ([0.30, 0.20, 0.15, 0.20, 0.05, 0.10], True, False),
}
LOSS_ARMS = {
    "field": [0.0, 0.0, 0.0, 0.0],
    "receiver": [0.05, 0.0, 0.0, 0.0],
    "phase": [0.05, 0.01, 0.0, 0.0],
    "local_spectrum": [0.05, 0.01, 0.02, 0.0],
    "energy": [0.05, 0.01, 0.02, 0.005],
}
LOSS_WEIGHT_KEYS = tuple(f"{name}_weight" for name in LOSS_NAMES)
STAGES = {
    64: {"target_height": 64, "target_width": 64, "global_size": 64,
         "learning_rate": 3e-4, "base_updates": 2000, "aux_updates": 1000,
         "query_chunk_size": 256},
    128: {"target_height": 128, "target_width": 128, "global_size": 100,
          "learning_rate": 2e-4, "optimizer_updates": 3000, "query_chunk_size": 512},
    400: {"target_height": 400, "target_width": 400, "global_size": 100,
          "learning_rate": 1e-4, "optimizer_updates": 6000, "query_chunk_size": 2048},
}
B0_MODEL = {
    "name": "factorized_fno", "in_features": 3, "out_channels": 1,
    "spatial_modes_x": 48, "spatial_modes_z": 48, "spatial_width": 8,
    "spatial_layers": 1, "temporal_width": 8, "temporal_kernel": 3,
    "temporal_layers": 1, "num_groups": 4, "head_hidden": 32,
    "temporal_chunk_size": 4, "activation_checkpointing": True,
}


def configure_legacy_b0(cfg: dict[str, Any], *, smoke: bool, run_name: str) -> None:
    """Add the concrete schema consumed by scripts/train_pino.py."""
    data_path = cfg["data"]["path"]
    cfg["data"].update({
        "raw_hdf5": data_path, "schema_mode": "explicit_after_inspection",
        "velocity_key": "nu", "wavefield_key": "tensor", "source_map_key": "source_mask",
        "source_position_x_key": "source_x_idx", "source_position_z_key": "source_z_idx",
        "wavelet_key": "wavelet", "frequency_key": "source_frequency_hz",
        "amplitude_key": "source_amplitude", "time_key": "t-coordinate",
        "x_key": "x-coordinate", "z_key": "y-coordinate", "dx_key": "dx",
        "velocity_axes": ["sample", "x", "z"],
        "wavefield_axes": ["sample", "time", "x", "z"],
        "source_map_axes": ["sample", "x", "z"], "source_position_units": "index",
        "input_features": ["time", "source_map", "velocity"],
        "split_seed": 20260714, "split": [0.8, 0.1, 0.1],
        "max_samples": None, "open_hdf5_per_sample": False, "num_workers": 0,
        "pin_memory": True, "persistent_workers": False, "shuffle_train": True,
    })
    cfg["sampling"]["spatial_method"] = "bilinear_antialias"
    cfg["train"].update({
        "device": "cuda", "epochs": 1 if smoke else 6,
        "learning_rate": 1e-4, "weight_decay": 1e-4, "scheduler": "cosine",
        "log_every_steps": 1, "checkpoint_dir": f"artifacts/{run_name}/checkpoints",
        "log_dir": f"artifacts/{run_name}/logs",
        "max_train_batches": 1 if smoke else None,
        "max_val_batches": 1 if smoke else None,
        "spatial_importance_sampling": {
            "enabled": True,
            "pixels_per_sample": 2048,
            "uniform_fraction": 1.0,
            "reweight_loss": True,
        },
    })
    cfg["loss"] = {"relative_l2_weight": 1.0, "mse_weight": 0.1, "eps": 1e-8}
    cfg["normalization"].update({"reuse_stats": True, "eps": 1e-6})

CONFIG_FILENAMES = (
    "factorized_fno_400x400x160_b0.yaml",
    "factorized_fno_400x400x160_b0_smoke.yaml",
    "ais_mqfno_64x160_b1_uniform.yaml",
    "ais_mqfno_64x160_s_static_hh.yaml",
    "ais_mqfno_64x160_a_residual_uncorrected.yaml",
    "ais_mqfno_64x160_ai_adaptive_hh.yaml",
    "ais_mqfno_64x160_ai_receiver.yaml",
    "ais_mqfno_64x160_ai_phase.yaml",
    "ais_mqfno_64x160_ai_local_spectrum.yaml",
    "ais_mqfno_64x160_ai_energy.yaml",
    "ais_mqfno_128x160_ai_adaptive_hh.yaml",
    "ais_mqfno_400x160_ai_adaptive_hh.yaml",
)


def label_sites_per_update(loss: dict[str, Any]) -> int:
    return 2048 + (50 if float(loss["receiver_weight"]) > 0 else 0) + (
        4 * 17 * 17 if float(loss["local_spectrum_weight"]) > 0 else 0
    )


def _set_loss(cfg: dict[str, Any], weights: list[float]) -> None:
    for name, value in zip(LOSS_NAMES, weights, strict=True):
        cfg["loss"][f"{name}_weight"] = value


def canonical_loss_profiles(selected_loss: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return phase overrides, including the exact loss frozen after selection."""
    profiles = {
        name: dict(zip(LOSS_WEIGHT_KEYS, weights, strict=True))
        for name, weights in LOSS_ARMS.items()
    }
    profiles["selected_frozen_loss"] = deepcopy(selected_loss)
    return profiles


def _set_stage(cfg: dict[str, Any], stage: int) -> None:
    values = STAGES[stage]
    cfg["sampling"].update({key: value for key, value in values.items() if key in {
        "target_height", "target_width", "global_size"}})
    cfg["train"]["learning_rate"] = values["learning_rate"]
    cfg["train"]["query_chunk_size"] = values["query_chunk_size"]


def _record_schedule(cfg: dict[str, Any], updates: int) -> None:
    sites = label_sites_per_update(cfg["loss"])
    cfg["experiment"]["label_site_schedule"] = [[updates, sites]]
    cfg["experiment"]["full160_label_sites"] = updates * sites


def build_configs() -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    b0 = deepcopy(COMMON)
    _set_stage(b0, 400)
    b0["model"] = deepcopy(B0_MODEL)
    b0["train"]["phases"] = [{"name": "matched_budget_b0", "optimizer_updates": 12000,
                                "loss_profile": "field", "reset_optimizer": True,
                                "init_from": "random"}]
    _record_schedule(b0, 12000)
    configure_legacy_b0(
        b0, smoke=False,
        run_name="ais_mqfno_full160_native400_20260714/baselines/b0",
    )
    outputs[CONFIG_FILENAMES[0]] = b0
    smoke = deepcopy(b0)
    smoke["experiment"].update({"smoke": True, "optimizer_updates": 2,
                                 "physical_scene_draws": 2, "gpu_hour_budget": 0.1})
    smoke["train"].update({"query_sites_per_scene": 32, "query_chunk_size": 16})
    smoke["train"]["phases"][0]["optimizer_updates"] = 2
    configure_legacy_b0(smoke, smoke=True, run_name="ais_mqfno_b0_native400_smoke")
    smoke["experiment"]["label_site_schedule"] = [[2, 32]]
    smoke["experiment"]["full160_label_sites"] = 64
    outputs[CONFIG_FILENAMES[1]] = smoke

    for arm_name, (mixture, hh, diagnostic) in SAMPLER_ARMS.items():
        cfg = deepcopy(COMMON)
        _set_stage(cfg, 64)
        cfg["sampler"]["mixture"] = dict(zip(MIXTURE_NAMES, mixture, strict=True))
        cfg["loss"]["hh_reweight"] = hh
        cfg["experiment"]["diagnostic_only"] = diagnostic
        cfg["train"]["phases"] = [{"name": arm_name, "optimizer_updates": 3000,
                                    "loss_profile": "field", "reset_optimizer": True,
                                    "init_from": "random"}]
        _record_schedule(cfg, 3000)
        outputs[f"ais_mqfno_64x160_{arm_name}.yaml"] = cfg

    for loss_name in LOSS_NAMES:
        cfg = deepcopy(outputs["ais_mqfno_64x160_ai_adaptive_hh.yaml"])
        _set_loss(cfg, LOSS_ARMS[loss_name])
        cfg["train"]["phases"] = [{"name": f"aux_{loss_name}", "optimizer_updates": 3000,
                                    "loss_profile": loss_name, "reset_optimizer": True,
                                    "init_from": "external_checkpoint"}]
        _record_schedule(cfg, 3000)
        outputs[f"ais_mqfno_64x160_ai_{loss_name}.yaml"] = cfg

    for stage in (128, 400):
        cfg = deepcopy(outputs["ais_mqfno_64x160_ai_adaptive_hh.yaml"])
        _set_stage(cfg, stage)
        updates = STAGES[stage]["optimizer_updates"]
        cfg["train"]["phases"] = [{"name": f"selected_{stage}", "optimizer_updates": updates,
                                    "loss_profile": "selected_frozen_loss",
                                    "reset_optimizer": True, "init_from": "external_checkpoint"}]
        _record_schedule(cfg, updates)
        outputs[f"ais_mqfno_{stage}x160_ai_adaptive_hh.yaml"] = cfg
    for name, cfg in outputs.items():
        if name.startswith("ais_mqfno"):
            cfg["loss_profiles"] = canonical_loss_profiles(cfg["loss"])
    if tuple(outputs) != CONFIG_FILENAMES:
        raise AssertionError("internal config declaration order drifted")
    return outputs


def _bytes(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(payload, sort_keys=True).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        yaml.safe_load(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    configs = build_configs()
    drift = []
    for name, payload in configs.items():
        expected = _bytes(payload)
        path = args.output_dir / name
        if args.check:
            if not path.is_file() or path.read_bytes() != expected:
                drift.append(name)
        else:
            _atomic_write(path, expected)
    if drift:
        parser.error("generated config drift: " + ", ".join(drift))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
