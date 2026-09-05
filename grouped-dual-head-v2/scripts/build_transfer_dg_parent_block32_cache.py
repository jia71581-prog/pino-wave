#!/usr/bin/env python3
"""Cache a frozen pyramid parent's train-only coefficients for block correction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.coupled_pyramid_moe_wave import (  # noqa: E402
    PyramidMoECoupledWaveOperator,
    parameter_count,
)
from scripts.train_transfer_dg_coupled_mhc_muon_pilot import (  # noqa: E402
    BLOCK,
    model_prediction,
)
from scripts.train_transfer_dg_coupled_pyramid_moe_full_ddp4 import (  # noqa: E402
    DirectFullPretrainData,
)
from scripts.train_transfer_dg_phase_scatter64_full_ddp import (  # noqa: E402
    FullCollection,
    atomic_json,
    sha256,
)
from scripts.train_transfer_dg_wfp_e1 import CacheCollection  # noqa: E402
from scripts.train_transfer_dg_wfp_e1d import TravelCollection  # noqa: E402


PHYSICAL_SCALE = 1.0e-8
CONDITION_CHANNELS = 9


def public_conditioning(static: dict) -> np.ndarray:
    """Build public medium/source maps without reading any wavefield."""

    medium = np.asarray(static["medium"], dtype=np.float32)
    source_map = np.asarray(static["source_map"], dtype=np.float32)
    travel = np.asarray(static["travel"], dtype=np.float32)
    sx, sz, f0, t0 = (float(value) for value in static["parameters"])
    if medium.shape != (16, 221, 241) or source_map.shape != (201, 201):
        raise RuntimeError("unexpected Transfer-DG public feature shape")
    x = np.arange(201, dtype=np.float32) * 10.0
    z = np.arange(201, dtype=np.float32) * 10.0
    zz, xx = np.meshgrid(z, x, indexing="ij")
    constant_f0 = np.full_like(source_map, np.clip((f0 - 20.0) / 10.0, -2.0, 2.0))
    constant_t0 = np.full_like(source_map, np.clip((t0 - 0.10) / 0.05, -2.0, 2.0))
    condition = np.concatenate(
        (
            medium[:3, :201, 20:221],
            source_map[None],
            np.clip((xx - sx) / 2000.0, -1.2, 1.2)[None],
            np.clip((zz - sz) / 2000.0, -0.2, 1.2)[None],
            travel[None],
            constant_f0[None],
            constant_t0[None],
        ),
        axis=0,
    ).astype(np.float32)
    if condition.shape != (CONDITION_CHANNELS, 201, 201) or not np.isfinite(condition).all():
        raise RuntimeError("invalid public conditioning bundle")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-cache", type=Path, action="append", required=True)
    parser.add_argument("--base-cache", type=Path, action="append", required=True)
    parser.add_argument("--travel", type=Path, action="append", required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    if args.output_cache.exists() or args.output_dir.exists():
        raise FileExistsError("refusing to reuse parent block-cache outputs")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    staging = args.output_cache.with_name(f"{args.output_cache.name}.partial.{os.getpid()}")
    data = None
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        bound = {
            "cache_builder_sha256": Path(__file__),
            "parent_model_sha256": ROOT / "saved_time_phase_operator_v4/coupled_pyramid_moe_wave.py",
            "parent_base_model_sha256": ROOT / "saved_time_phase_operator_v4/coupled_mhc_wave.py",
            "full_manifest_sha256": args.full_manifest,
            "selection_manifest_sha256": args.selection_manifest,
            "parent_checkpoint_sha256": args.parent_checkpoint,
        }
        for key, path in bound.items():
            observed = sha256(path)
            if observed != bindings[key]:
                raise RuntimeError(f"parent block-cache binding drift: {key} {observed}")

        selection = json.loads(args.selection_manifest.read_text())
        if selection.get("split") != "train":
            raise RuntimeError("parent block cache accepts train records only")
        if selection.get("validation_opened") or selection.get("test_id_opened"):
            raise RuntimeError("selection manifest has an opened sealed split")
        if selection.get("selection_uses_wavefield_truth"):
            raise RuntimeError("record selection may not use wavefield truth")
        records = list(selection["records"])
        registered_count = int(prereg["data"]["record_count"])
        if len(records) != registered_count:
            raise RuntimeError("selection record count differs from preregistration")
        if args.max_records:
            expected = int(prereg["smoke"]["cache_max_records"])
            if args.max_records != expected:
                raise RuntimeError("unregistered cache max-record override")
            records = records[: args.max_records]

        full_manifest = json.loads(args.full_manifest.read_text())
        residual = FullCollection(args.residual_cache, full_manifest)
        base = CacheCollection(args.base_cache, full_manifest, expected_count=2800)
        travel = TravelCollection(args.travel, expected_count=2800)
        data = DirectFullPretrainData(residual, base, travel)
        position_by_id = {row[2]: index for index, row in enumerate(residual.records)}
        if any(row["sample_id"] not in position_by_id for row in records):
            raise RuntimeError("selection contains a sample absent from full train caches")

        checkpoint = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("validation_opened") or checkpoint.get("test_id_opened"):
            raise RuntimeError("parent checkpoint has an opened sealed split")
        if checkpoint.get("schema") != prereg["parent"]["schema"]:
            raise RuntimeError("parent checkpoint schema drift")
        if int(checkpoint.get("epoch", -1)) != int(prereg["parent"]["epoch"]):
            raise RuntimeError("parent checkpoint epoch drift")
        if int(checkpoint.get("update", -1)) != int(prereg["parent"]["update"]):
            raise RuntimeError("parent checkpoint update drift")
        device = torch.device("cuda")
        model = PyramidMoECoupledWaveOperator(use_mhc=True).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval().requires_grad_(False)
        if parameter_count(model) != int(prereg["parent"]["parameter_count"]):
            raise RuntimeError("parent parameter count drift")

        source_h5 = Path(selection["source_h5"])
        if sha256(source_h5) != bindings["source_h5_sha256"]:
            raise RuntimeError("source dataset binding drift")
        args.output_cache.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(source_h5, "r", swmr=True) as source, h5py.File(staging, "w") as output:
            output.attrs.update(
                {
                    "schema": "transfer_dg_parent_anchored_block_cache_v1",
                    "status": "building",
                    "split": "train",
                    "record_count": len(records),
                    "frequency_count": 64,
                    "time_count": int(source["time_s"].shape[0]),
                    "physical_scale": PHYSICAL_SCALE,
                    "condition_channels": CONDITION_CHANNELS,
                    "parent_checkpoint_sha256": bindings["parent_checkpoint_sha256"],
                    "selection_manifest_sha256": bindings["selection_manifest_sha256"],
                    "source_h5_sha256": bindings["source_h5_sha256"],
                    "offline_train_truth_cached": False,
                    "validation_opened": False,
                    "test_id_opened": False,
                }
            )
            string = h5py.string_dtype("utf-8")
            output.create_dataset(
                "sample_id", data=np.asarray([row["sample_id"] for row in records], dtype=object), dtype=string
            )
            output.create_dataset(
                "family", data=np.asarray([row["family"] for row in records], dtype=object), dtype=string
            )
            output.create_dataset(
                "role", data=np.asarray([row["role"] for row in records], dtype=object), dtype=string
            )
            output.create_dataset(
                "group_id", data=np.asarray([row["group_id"] for row in records], dtype=object), dtype=string
            )
            output.create_dataset(
                "source_index", data=np.asarray([row["source_index"] for row in records], dtype=np.int64)
            )
            parent_dataset = output.create_dataset(
                "parent_coefficients",
                shape=(len(records), 64, 2, 201, 201),
                dtype=np.float16,
                chunks=(1, 1, 2, 201, 201),
            )
            condition_dataset = output.create_dataset(
                "condition",
                shape=(len(records), CONDITION_CHANNELS, 201, 201),
                dtype=np.float16,
                chunks=(1, CONDITION_CHANNELS, 201, 201),
            )
            maximum_parent = 0.0
            for output_index, row in enumerate(records):
                sample_id = str(row["sample_id"])
                source_index = int(row["source_index"])
                source_sample = source["sample_id"][source_index]
                source_sha256 = source["sample_sha256"][source_index]
                source_group = source["group_id"][source_index]
                source_family = source["medium_type"][source_index]
                if isinstance(source_sample, bytes):
                    source_sample = source_sample.decode()
                if isinstance(source_sha256, bytes):
                    source_sha256 = source_sha256.decode()
                if isinstance(source_group, bytes):
                    source_group = source_group.decode()
                if isinstance(source_family, bytes):
                    source_family = source_family.decode()
                source_split = source["split"][source_index]
                if isinstance(source_split, bytes):
                    source_split = source_split.decode()
                if (
                    source_sample != sample_id
                    or source_split != "train"
                    or source_sha256 != str(row["sample_sha256"])
                    or source_group != str(row["group_id"])
                    or source_family != str(row["family"])
                ):
                    raise RuntimeError("source index/sample/group/family/hash/split binding drift")
                position = position_by_id[sample_id]
                coefficients = []
                for start in range(0, 64, BLOCK):
                    batch = data.block(position, start, device)
                    with torch.inference_mode():
                        prediction, _ = model_prediction(model, batch)
                    coefficients.append(prediction.float().cpu())
                parent = torch.cat(coefficients, dim=0).numpy()
                if parent.shape != (64, 2, 201, 201) or not np.isfinite(parent).all():
                    raise RuntimeError("non-finite or malformed parent coefficients")
                if float(np.max(np.abs(parent))) >= np.finfo(np.float16).max:
                    raise FloatingPointError("parent coefficients exceed float16 cache range")
                parent_dataset[output_index] = parent.astype(np.float16)
                condition_dataset[output_index] = public_conditioning(
                    residual.static(position)
                ).astype(np.float16)
                maximum_parent = max(maximum_parent, float(np.max(np.abs(parent))))
                if output_index % 4 == 0 or output_index + 1 == len(records):
                    event = {
                        "event": "cache_progress",
                        "record": output_index + 1,
                        "of": len(records),
                        "sample_id": sample_id,
                        "maximum_parent": maximum_parent,
                    }
                    print(json.dumps(event, sort_keys=True), flush=True)
            output.attrs["maximum_parent"] = maximum_parent
            output.attrs["status"] = "complete"
            output.flush()
        os.replace(staging, args.output_cache)
        identity = {
            "schema": "transfer_dg_parent_anchored_block_cache_identity_v1",
            "status": "complete",
            "record_count": len(records),
            "cache": str(args.output_cache.resolve()),
            "cache_sha256": sha256(args.output_cache),
            "parent_checkpoint": str(args.parent_checkpoint.resolve()),
            "parent_checkpoint_sha256": bindings["parent_checkpoint_sha256"],
            "selection_manifest": str(args.selection_manifest.resolve()),
            "selection_manifest_sha256": bindings["selection_manifest_sha256"],
            "source_h5": str(source_h5.resolve()),
            "source_h5_sha256": bindings["source_h5_sha256"],
            "future_truth_scope": "No wavefield read; cache contains parent predictions and public conditioning only",
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(identity, args.output_dir / "run_identity.json")
        atomic_json(
            {"status": "complete", "cache": str(args.output_cache.resolve()), "cache_sha256": identity["cache_sha256"]},
            terminal,
        )
        print(json.dumps(json.loads(terminal.read_text()), sort_keys=True), flush=True)
        return 0
    except Exception as error:
        if staging.exists():
            staging.unlink()
        atomic_json(
            {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()},
            terminal,
        )
        raise
    finally:
        if data is not None:
            data.close()


if __name__ == "__main__":
    raise SystemExit(main())
