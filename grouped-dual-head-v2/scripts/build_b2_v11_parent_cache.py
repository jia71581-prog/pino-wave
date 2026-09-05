#!/usr/bin/env python3
"""Render immutable frozen-parent predictions for B2-v11 train-only meta tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.train_b2_snapshot_ic import FrameConditionedPropagator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _load_parent(checkpoint: dict, device: torch.device) -> tuple[torch.nn.Module, str]:
    identity = checkpoint["identity"]
    conditioning_key = str(identity["conditioning_key"])
    model = FrameConditionedPropagator(
        state_channels=8,
        cond_channels=int(identity["cond_channels"]),
        width=int(identity.get("width", 64)),
        spectral_rank=int(identity.get("spectral_rank", 32)),
        modes=24,
        depth=4,
        gate_init=1.0,
        activation_checkpointing=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, conditioning_key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_cache.exists() or args.output_dir.exists():
        raise FileExistsError("refusing to reuse B2-v11 cache outputs")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    staging = args.output_cache.with_name(
        f"{args.output_cache.name}.partial.{os.getpid()}"
    )
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        for path, key in (
            (Path(__file__), "builder_sha256"),
            (args.checkpoint, "checkpoint_sha256"),
            (args.source_cache, "source_cache_sha256"),
            (args.manifest, "manifest_sha256"),
        ):
            if _sha256(path) != bindings[key]:
                raise RuntimeError(f"binding drift: {path}")
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("split") != "train":
            raise RuntimeError("B2-v11 parent cache accepts train records only")
        if manifest.get("validation_opened") or manifest.get("test_id_opened"):
            raise RuntimeError("sealed split flag is open")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        device = torch.device("cuda")
        model, conditioning_key = _load_parent(checkpoint, device)
        with h5py.File(args.source_cache, "r", swmr=True) as source:
            if source.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
                raise RuntimeError("source cache is not B2-v5 causal")
            if source.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError("source cache/manifest drift")
            if conditioning_key not in source:
                raise RuntimeError(f"conditioning dataset absent: {conditioning_key}")
            count, total, height, width = source["base_seq"].shape
            with h5py.File(staging, "w") as output:
                output.attrs["schema"] = "b2_v11_parent_prediction_cache_v1"
                output.attrs["status"] = "building"
                output.attrs["checkpoint_sha256"] = _sha256(args.checkpoint)
                output.attrs["source_cache_sha256"] = _sha256(args.source_cache)
                output.attrs["manifest_selection_sha256"] = manifest["selection_sha256"]
                output.attrs["conditioning_key"] = conditioning_key
                output.create_dataset(
                    "sample_id", data=source["sample_id"][:], dtype=h5py.string_dtype()
                )
                output.create_dataset(
                    "family", data=source["family"][:], dtype=h5py.string_dtype()
                )
                parent_dataset = output.create_dataset(
                    "parent",
                    shape=(count, total, height, width),
                    dtype=np.float16,
                )
                for index in range(count):
                    base = torch.from_numpy(
                        source["base_seq"][index : index + 1].astype(np.float32)
                    )[:, :, None].to(device)
                    observed = torch.from_numpy(
                        source["target"][index : index + 1, :8].astype(np.float32)
                    ).to(device)
                    conditioning = torch.from_numpy(
                        source[conditioning_key][index : index + 1].astype(np.float32)
                    ).to(device)
                    with torch.inference_mode():
                        prediction = model.forward_anchored(
                            base,
                            conditioning,
                            initial_state=observed,
                        )
                    parent_dataset[index] = prediction[0, :, 0].float().cpu().numpy().astype(np.float16)
                    if index % 20 == 0:
                        print(
                            json.dumps(
                                {"event": "parent_cache_progress", "record": index, "of": count}
                            ),
                            flush=True,
                        )
                output.attrs["status"] = "complete"
                output.flush()
        os.replace(staging, args.output_cache)
        identity = {
            "schema": "b2_v11_parent_cache_identity_v1",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "source_cache": str(args.source_cache),
            "source_cache_sha256": _sha256(args.source_cache),
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "output_cache": str(args.output_cache),
            "output_cache_sha256": _sha256(args.output_cache),
            "builder_sha256": _sha256(Path(__file__)),
            "conditioning_key": conditioning_key,
            "future_truth_scope": "parent inference uses only the eight allowed anchors",
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        _atomic_json(
            {
                "status": "complete",
                "cache": str(args.output_cache),
                "cache_sha256": identity["output_cache_sha256"],
            },
            terminal,
        )
        print(json.dumps(json.loads(terminal.read_text()), indent=2))
        return 0
    except Exception as error:
        import traceback

        _atomic_json(
            {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()},
            terminal,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
