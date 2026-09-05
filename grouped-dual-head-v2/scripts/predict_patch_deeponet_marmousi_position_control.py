#!/usr/bin/env python
"""Input-only Patch-DeepONet predictions for the fixed-19-Hz position panel.

This command intentionally has no reference-generation or scoring mode. It reads
only velocity/source/coordinate inputs, seals 240 predictions with the same schema
as the proposed model, and leaves reference reuse to the existing sealed scorer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fno_acoustic.data_generation.source import bilinear_point_source
from grouped_ufno_mionet_v3.data.index import build_manifest
from grouped_ufno_mionet_v3.model.travel_time import (
    dense_query_coordinates,
    straight_ray_travel_time,
)
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
from patch_deeponet_baseline.features import (
    build_query_descriptors,
    build_static_features,
)
from patch_deeponet_baseline.model import PatchDeepONet, PatchDeepONetConfig
from saved_time_phase_operator_v4.evaluation import sha256_file, time_axis_sha256
from scripts.evaluate_marmousi_source_position_control import (
    PREDICTION_SCHEMA,
    _atomic_json,
    _atomic_npz,
    load_locked_inputs,
    load_protocol,
    locked_cases,
)


IDENTITY_SCHEMA = "patch_deeponet_run_identity_v1"
CHECKPOINT_SCHEMA = "patch_deeponet_checkpoint_v1"


def _load_baseline(
    checkpoint_path: Path,
    identity_path: Path,
    *,
    expected_manifest_digest: str,
    device: torch.device,
) -> tuple[PatchDeepONet, dict[str, Any], Mapping[str, Any]]:
    identity = json.loads(identity_path.read_text(encoding="utf8"))
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise ValueError("unexpected Patch-DeepONet run identity schema")
    if identity.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("Patch-DeepONet identity and position manifest disagree")
    if identity.get("selection_split") != "train":
        raise ValueError("Patch-DeepONet checkpoint was not selected on train only")
    config = PatchDeepONetConfig(**identity["model_config"])
    model = PatchDeepONet(config)
    if model.parameter_count() != int(identity["parameter_count"]):
        raise ValueError("Patch-DeepONet identity parameter count changed")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Patch-DeepONet checkpoint must be a mapping")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected Patch-DeepONet checkpoint schema")
    if checkpoint.get("run_digest") != identity.get("run_digest"):
        raise ValueError("Patch-DeepONet checkpoint and run identity disagree")
    if checkpoint.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("Patch-DeepONet checkpoint and position manifest disagree")
    if checkpoint.get("selection_split") != "train":
        raise ValueError("Patch-DeepONet checkpoint was not selected on train only")
    if int(checkpoint.get("epoch", -1)) < 0 or int(checkpoint.get("global_step", -1)) < 0:
        raise ValueError("Patch-DeepONet checkpoint counters are invalid")
    if "model_state" not in checkpoint:
        raise ValueError("Patch-DeepONet checkpoint has no model state")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, identity, checkpoint


@torch.inference_mode()
def _dense_prediction(
    model: PatchDeepONet,
    normalizer: PhysicalNormalizer,
    velocity_mps: np.ndarray,
    source_parameters: list[float],
    source_map: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    *,
    device: torch.device,
    time_block: int,
    query_chunk: int,
) -> np.ndarray:
    if time_block <= 0 or query_chunk <= 0:
        raise ValueError("time block and query chunk must be positive")
    velocity = torch.from_numpy(velocity_mps)[None, None].to(device)
    source = torch.tensor(source_parameters, dtype=torch.float32, device=device)[None]
    source_n = normalizer.encode_source(source)
    source_map_t = torch.from_numpy(source_map)[None, None].to(device)
    x_t = torch.from_numpy(x_m.astype(np.float32)).to(device)
    z_t = torch.from_numpy(z_m.astype(np.float32)).to(device)
    coords_xy = dense_query_coordinates(x_t, z_t, records=1)
    travel = straight_ray_travel_time(
        velocity,
        source[:, :2],
        coords_xy,
        x_extent_m=(0.0, 2000.0),
        z_extent_m=(0.0, 2000.0),
        samples=12,
    )
    static = build_static_features(
        normalizer.encode_velocity(velocity),
        source_map_t,
        travel.seconds.reshape(1, len(z_m), len(x_m)),
        velocity,
        domain_t_s=1.0,
    )
    encoded = model.encode_static(static)
    outputs: list[torch.Tensor] = []
    for time_start in range(0, len(time_s), int(time_block)):
        times = torch.from_numpy(
            time_s[time_start : time_start + int(time_block)].astype(np.float32)
        ).to(device)
        count = len(times)
        points = coords_xy.shape[1]
        xyz = torch.cat(
            (
                coords_xy[:, None].expand(-1, count, -1, -1),
                times[None, :, None, None].expand(1, -1, points, 1),
            ),
            dim=-1,
        ).reshape(1, count * points, 3)
        travel_block = travel.seconds[:, None].expand(-1, count, -1).reshape(1, -1)
        descriptors = build_query_descriptors(
            xyz,
            source_n,
            source,
            travel_block,
            domain_x_m=2000.0,
            domain_z_m=2000.0,
            domain_t_s=1.0,
        )
        pieces = [
            model.query_encoded(static, encoded, descriptors[:, start : start + query_chunk])
            for start in range(0, descriptors.shape[1], int(query_chunk))
        ]
        block = torch.cat(pieces, dim=1).reshape(1, count, len(z_m), len(x_m))
        # Both methods know the homogeneous Dirichlet free-surface boundary.
        block[..., 0, :] = 0.0
        outputs.append(block)
    normalized = torch.cat(outputs, dim=1)
    prediction = normalizer.decode_pressure(normalized, source[:, 4]).squeeze(0)
    result = prediction.cpu().numpy().astype(np.float32)
    expected_shape = (len(time_s), len(z_m), len(x_m))
    if result.shape != expected_shape or not np.isfinite(result).all():
        raise RuntimeError("Patch-DeepONet produced an invalid complete transient")
    return result


@torch.inference_mode()
def predict(
    *,
    base_config_path: Path,
    checkpoint_path: Path,
    identity_path: Path,
    protocol_path: Path,
    output_dir: Path,
    device: torch.device,
    time_block: int,
    query_chunk: int,
) -> dict[str, Any]:
    manifest_path = output_dir / "prediction_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("prediction seal already exists; refusing overwrite")
    base = yaml.safe_load(base_config_path.read_text(encoding="utf8"))
    source_h5 = Path(base["data"]["source_h5"])
    manifest = build_manifest(source_h5)
    normalizer = PhysicalNormalizer.from_dict(
        json.loads(Path(base["data"]["normalization_json"]).read_text(encoding="utf8")),
        expected_manifest=manifest.digest,
    )
    protocol = load_protocol(protocol_path)
    cases = locked_cases(protocol)
    inputs, time_s, x_m, z_m = load_locked_inputs(source_h5, manifest, protocol)
    model, identity, checkpoint = _load_baseline(
        checkpoint_path,
        identity_path,
        expected_manifest_digest=manifest.digest,
        device=device,
    )
    by_rank = {rank: [row for row in cases if row["slice_rank"] == rank] for rank in inputs}
    input_entries: list[dict[str, Any]] = []
    prediction_entries: list[dict[str, Any]] = []
    for rank in sorted(inputs):
        item = inputs[rank]
        input_path = output_dir / "inputs" / f"marm_r{rank:02d}_velocity.npz"
        _atomic_npz(input_path, velocity_mps=item["velocity_mps"], time_s=time_s, x_m=x_m, z_m=z_m)
        input_entries.append(
            {
                "slice_rank": rank,
                "source_index": item["source_index"],
                "source_sample_id": item["sample_id"],
                "source_group_id": item["group_id"],
                "velocity_array_sha256": item["velocity_sha256"],
                "input_path": str(input_path.resolve()),
                "input_file_sha256": sha256_file(input_path),
            }
        )
        for case in by_rank[rank]:
            source_values = case["source_parameters"]
            point = bilinear_point_source(
                float(source_values[0]),
                float(source_values[1]),
                nx=len(x_m),
                nz=len(z_m),
                dx_m=float(x_m[1] - x_m[0]),
                dz_m=float(z_m[1] - z_m[0]),
                centering="node",
            )
            prediction = _dense_prediction(
                model,
                normalizer,
                item["velocity_mps"],
                source_values,
                point.source_map,
                time_s,
                x_m,
                z_m,
                device=device,
                time_block=time_block,
                query_chunk=query_chunk,
            )
            prediction_path = output_dir / "predictions" / f"{case['record_id']}.npz"
            _atomic_npz(
                prediction_path,
                prediction_tzx=prediction,
                source_map_zx=point.source_map,
                source_parameters=np.asarray(source_values, dtype=np.float64),
            )
            prediction_entries.append(
                {**case, "prediction_path": str(prediction_path.resolve()), "prediction_sha256": sha256_file(prediction_path)}
            )
    prediction_entries.sort(key=lambda row: (row["slice_rank"], row["case_id"]))
    payload = {
        "schema": PREDICTION_SCHEMA,
        "status": "complete",
        "truth_wavefield_access": False,
        "source_frequency_varied": False,
        "input_hdf5_datasets_accessed": ["time_s", "velocity_mps", "x_m", "z_m"],
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "config": str(base_config_path.resolve()),
        "config_file_sha256": sha256_file(base_config_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "checkpoint_identity": str(identity_path.resolve()),
        "checkpoint_identity_sha256": sha256_file(identity_path),
        "model_config_digest": str(identity["run_digest"]),
        "manifest_digest": manifest.digest,
        "time_axis_sha256": time_axis_sha256(time_s),
        "fixed_source_parameters": protocol["fixed_source_parameters"],
        "model_class": "parameter_matched_patch_deeponet",
        "model_parameter_count": model.parameter_count(),
        "travel_feature_rule": "marmousi=straight_ray12, matching the frozen training cache",
        "inputs": input_entries,
        "records": prediction_entries,
    }
    if len(prediction_entries) != 240:
        raise AssertionError("Patch-DeepONet prediction record count changed")
    _atomic_json(payload, manifest_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-identity", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time-block", type=int, default=4)
    parser.add_argument("--query-chunk", type=int, default=16384)
    args = parser.parse_args()
    payload = predict(
        base_config_path=args.base_config,
        checkpoint_path=args.checkpoint,
        identity_path=args.checkpoint_identity,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        device=torch.device(args.device),
        time_block=args.time_block,
        query_chunk=args.query_chunk,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["predict", "_dense_prediction", "_load_baseline"]
