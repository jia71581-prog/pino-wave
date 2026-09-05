#!/usr/bin/env python3
"""Generate exactly eight preregistered AIS normalization-v2 candidates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILENAMES = (
    "n0_norm.yaml",
    "n1_wide.yaml",
    "n2_spatial.yaml",
    "n3_temporal.yaml",
    "n4_large_local.yaml",
    "n5_multi_local.yaml",
    "n6_dispersion.yaml",
    "n7_multi_dispersion.yaml",
)

SPLIT_MANIFEST = (
    "artifacts/hybrid_drp_pino/fno_continue_pretrain_optimizer_safe/splits.json"
)
SPLIT_MANIFEST_SHA256 = (
    "d56e6eb80e3a0c0f2eff95bc9e91b9b3b69e2c49ca2b545de75ff6627563df8c"
)
NORMALIZATION_STATS = "artifacts/pino_adaptation/normalization_stats.json"
NORMALIZATION_STATS_SHA256 = (
    "aaeffb678e45b233c2a4e437904cb50f27ee6245bf7c664e6d5e0a6404b8aea4"
)
CONFIG_DIR = "configs/ais_zero_collapse_v2"
REGISTRATION_DIR = f"{CONFIG_DIR}/registration"
REGISTRATION_FILENAMES = (
    "gate_o_sample_0002_sites_2048.json",
    "fixed_validation_sites_2048.json",
)
GATE_O_SAMPLE_ID = 2

BASE_MODEL: dict[str, Any] = {
    "name": "ais_mqfno",
    "global_in_features": 6,
    "native_in_channels": 5,
    "global_max_size": 100,
    "spatial_width": 24,
    "spatial_modes": 24,
    "spatial_layers": 3,
    "temporal_modes": 32,
    "local_dim": 16,
    "fusion_dim": 32,
    "halo_size": 17,
    "local_encoder_kind": "single",
    "dispersion_head": "none",
    "activation_checkpointing": True,
}

VARIANTS: dict[str, dict[str, Any]] = {
    "n0_norm.yaml": {},
    "n1_wide.yaml": {"spatial_width": 32, "local_dim": 24, "fusion_dim": 48},
    "n2_spatial.yaml": {"spatial_width": 32, "spatial_modes": 32},
    "n3_temporal.yaml": {"temporal_modes": 48, "fusion_dim": 48},
    "n4_large_local.yaml": {"halo_size": 25, "local_dim": 24},
    "n5_multi_local.yaml": {
        "local_encoder_kind": "multiscale_9_25",
        "halo_size": 25,
        "local_dim": 24,
        "fusion_dim": 48,
    },
    "n6_dispersion.yaml": {"dispersion_head": "phase_residual_24"},
    "n7_multi_dispersion.yaml": {
        "local_encoder_kind": "multiscale_9_25",
        "halo_size": 25,
        "local_dim": 24,
        "fusion_dim": 48,
        "dispersion_head": "phase_residual_24",
    },
}


def _fixed_site_indices() -> list[int]:
    """Match the trainer's unique rounded-linspace registration on a 64² grid."""

    total = 64 * 64
    count = 2048
    indices = [round(index * (total - 1) / (count - 1)) for index in range(count)]
    if len(indices) != count or len(set(indices)) != count:
        raise AssertionError("fixed site registration must contain 2048 unique sites")
    return indices


def _registered_splits() -> dict[str, list[int]]:
    path = PROJECT_ROOT / SPLIT_MANIFEST
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPLIT_MANIFEST_SHA256:
        raise ValueError("registered split manifest SHA-256 drifted")
    payload = json.loads(raw)
    splits = {name: payload[name] for name in ("train", "val", "test")}
    if GATE_O_SAMPLE_ID not in splits["train"]:
        raise ValueError("Gate O sample must belong to the registered train split")
    if GATE_O_SAMPLE_ID in splits["val"] or GATE_O_SAMPLE_ID in splits["test"]:
        raise ValueError("Gate O sample must not belong to val or test")
    return splits


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def build_registration_manifests() -> dict[str, dict[str, Any]]:
    """Return the two canonical fixed-site registrations bound to the split."""

    splits = _registered_splits()
    common = {
        "schema": "ais_fixed_spatial_sites",
        "schema_version": 1,
        "split_manifest": SPLIT_MANIFEST,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "grid": {
            "height": 64,
            "width": 64,
            "index_order": "row_major_x_then_z",
        },
        "site_count": 2048,
        "site_indices": FIXED_SITE_INDICES,
        "selection_strategy": "rounded_linspace_including_domain_endpoints",
    }
    return {
        REGISTRATION_FILENAMES[0]: {
            **deepcopy(common),
            "purpose": "gate_o_train_representability",
            "split": "train",
            "sample_id": GATE_O_SAMPLE_ID,
        },
        REGISTRATION_FILENAMES[1]: {
            **deepcopy(common),
            "purpose": "fixed_validation_sites",
            "split": "val",
            "sample_ids": list(splits["val"]),
            "sites_shared_across_scenes": True,
        },
    }


FIXED_SITE_INDICES = _fixed_site_indices()
REGISTRATION_MANIFESTS = build_registration_manifests()
GATE_O_SITE_MANIFEST_SHA256 = hashlib.sha256(
    _json_bytes(REGISTRATION_MANIFESTS[REGISTRATION_FILENAMES[0]])
).hexdigest()
VALIDATION_SITE_MANIFEST_SHA256 = hashlib.sha256(
    _json_bytes(REGISTRATION_MANIFESTS[REGISTRATION_FILENAMES[1]])
).hexdigest()

COMMON: dict[str, Any] = {
    "seed": 20260714,
    "data": {
        "path": "/home/jiayh/Data/data/pino.hdf5",
        "split_manifest": SPLIT_MANIFEST,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "tensor_axes": ["sample", "time", "x", "z"],
    },
    "normalization": {
        "contract": "ais_normalization_v2",
        "stats_path": NORMALIZATION_STATS,
        "stats_sha256": NORMALIZATION_STATS_SHA256,
        "static_features": [
            "v_hat",
            "source_hat",
            "dv_dxi",
            "dv_dzeta",
            "slow_contrast",
        ],
        "time_feature": "tau",
        "target": "u_hat",
    },
    "sampling": {
        "target_height": 64,
        "target_width": 64,
        "global_size": 64,
        "max_time_steps": 160,
    },
    "receiver": {
        "z_m": [20.0],
        "x_start_m": 0.0,
        "x_stop_m": 1960.0,
        "x_stride_m": 40.0,
    },
    "train": {
        "batch_size": 1,
        "query_sites_per_scene": 2048,
        "receiver_sites_per_scene": 50,
        "patch_centers_per_scene": 4,
        "patch_size": 17,
        "query_chunk_size": 256,
        "optimizer": "adamw",
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "grad_clip": 1.0,
        "amp": False,
        "scheduler": "cosine_by_update",
        "phases": [
            {
                "name": "screen_main",
                "optimizer_updates": 3000,
                "reset_optimizer": True,
                "init_from": "random",
            }
        ],
    },
    "loss": {
        "hh_reweight": True,
        "late_time_weights": [1.0, 1.0, 1.0, 1.25],
        "receiver_weight": 0.0,
        "phase_weight": 0.0,
        "local_spectrum_weight": 0.0,
        "energy_weight": 0.0,
        "pde_weight": 0.0,
        "drp_weight": 0.0,
    },
    "sampler": {
        "tile_grid": [16, 16],
        "ema_momentum": 0.9,
        "lag_epochs": 1,
        "residual_power": 1.0,
        "min_probability": 1.0e-12,
        "mixture": {
            "uniform": 0.30,
            "interface": 0.20,
            "source_wavefront": 0.15,
            "residual": 0.20,
            "edge": 0.05,
            "receiver": 0.10,
        },
    },
    "experiment": {
        "optimizer_updates": 3000,
        "physical_scene_draws": 3000,
        "query_unit": "spatial_site_full_trace",
        "label_site_schedule": [[3000, 2048]],
        "full160_label_sites": 6_144_000,
        "diagnostic_only": False,
    },
    "screen": {
        "gate_o_sample_id": GATE_O_SAMPLE_ID,
        "gate_o_site_manifest": (
            f"{REGISTRATION_DIR}/{REGISTRATION_FILENAMES[0]}"
        ),
        "gate_o_site_manifest_sha256": GATE_O_SITE_MANIFEST_SHA256,
        "gate_o_fixed_unique_sites": 2048,
        "validation_site_manifest": (
            f"{REGISTRATION_DIR}/{REGISTRATION_FILENAMES[1]}"
        ),
        "validation_site_manifest_sha256": VALIDATION_SITE_MANIFEST_SHA256,
        "validation_sites_per_scene": 2048,
        "scene_order": {
            "source": "split_manifest_train_order",
            "shuffle": False,
        },
        "budgets": {
            "gate_o": {
                "total_optimizer_updates": 400,
                "initialization": "restart_from_registered_seed",
                "separate_output": True,
            },
            "h1": {
                "total_optimizer_updates": 600,
                "additional_optimizer_updates": 600,
                "initialization": "restart_from_registered_seed",
                "reuse_gate_o_weights": False,
            },
            "h2": {
                "total_optimizer_updates": 1500,
                "additional_optimizer_updates": 900,
                "resume_from": "h1_update_600_last",
            },
            "h3": {
                "total_optimizer_updates": 3000,
                "additional_optimizer_updates": 1500,
                "resume_from": "h2_update_1500_last",
            },
        },
    },
}


def build_v2_configs() -> dict[str, dict[str, Any]]:
    """Return fresh canonical payloads for exactly N0 through N7."""

    if tuple(VARIANTS) != CONFIG_FILENAMES:
        raise AssertionError("AIS v2 candidate declaration order drifted")
    outputs: dict[str, dict[str, Any]] = {}
    for filename in CONFIG_FILENAMES:
        config = deepcopy(COMMON)
        config["model"] = {**deepcopy(BASE_MODEL), **deepcopy(VARIANTS[filename])}
        config["screen"]["candidate_id"] = filename.split("_", 1)[0].upper()
        outputs[filename] = config
    if len(outputs) != 8:
        raise AssertionError("AIS v2 screen requires exactly eight candidates")
    return outputs


def _bytes(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(payload, sort_keys=True).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / CONFIG_DIR,
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    configs = build_v2_configs()
    registrations = build_registration_manifests()
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_yaml = {Path(name) for name in configs}
    actual_yaml = {
        path.relative_to(args.output_dir) for path in args.output_dir.rglob("*.yaml")
    }
    extra_yaml = actual_yaml - expected_yaml
    expected_json = {Path("registration") / name for name in registrations}
    actual_json = {
        path.relative_to(args.output_dir) for path in args.output_dir.rglob("*.json")
    }
    extra_json = actual_json - expected_json
    if extra_yaml or extra_json:
        extras = sorted(str(path) for path in extra_yaml | extra_json)
        parser.error("unexpected AIS v2 artifacts: " + ", ".join(extras))

    drift: list[str] = []
    for filename, payload in configs.items():
        expected = _bytes(payload)
        path = args.output_dir / filename
        if args.check:
            if not path.is_file() or path.read_bytes() != expected:
                drift.append(filename)
        else:
            _atomic_write(path, expected)
    for filename, payload in registrations.items():
        expected = _json_bytes(payload)
        path = args.output_dir / "registration" / filename
        if args.check:
            if not path.is_file() or path.read_bytes() != expected:
                drift.append(f"registration/{filename}")
        else:
            _atomic_write(path, expected)
    if drift:
        parser.error("generated config drift: " + ", ".join(drift))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
