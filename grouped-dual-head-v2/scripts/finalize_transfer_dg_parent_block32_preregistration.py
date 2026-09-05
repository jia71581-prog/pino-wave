#!/usr/bin/env python3
"""Bind a completed parent cache into an immutable block-32 training record."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import h5py


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_transfer_dg_phase_scatter64_full_ddp import atomic_json, sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-preregistration", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a training preregistration")
    specification = json.loads(args.cache_preregistration.read_text())
    if specification.get("schema") != "transfer_dg_parent_anchored_block32_cache_preregistration_v1":
        raise RuntimeError("unexpected cache preregistration schema")
    if specification.get("validation_opened") or specification.get("test_id_opened"):
        raise RuntimeError("cache preregistration has an opened sealed split")
    for key, path in {
        "preregistration_finalizer_sha256": Path(__file__),
        "trainer_sha256": ROOT / "scripts/train_transfer_dg_parent_anchored_block32.py",
        "corrector_sha256": ROOT / "saved_time_phase_operator_v4/parent_anchored_block.py",
    }.items():
        if sha256(path) != specification["bindings"][key]:
            raise RuntimeError(f"training code binding drift before cache finalization: {key}")
    with h5py.File(args.cache, "r", swmr=True) as cache:
        if cache.attrs.get("schema", "") != "transfer_dg_parent_anchored_block_cache_v1":
            raise RuntimeError("unexpected completed-cache schema")
        if cache.attrs.get("status", "") != "complete" or cache.attrs.get("split", "") != "train":
            raise RuntimeError("cache is incomplete or not train-only")
        if cache.attrs.get("validation_opened") or cache.attrs.get("test_id_opened"):
            raise RuntimeError("cache has an opened sealed split")
        if int(cache.attrs["record_count"]) != int(specification["data"]["record_count"]):
            raise RuntimeError("cache record count differs from specification")
        if str(cache.attrs["parent_checkpoint_sha256"]) != specification["parent"]["checkpoint_sha256"]:
            raise RuntimeError("cache parent checkpoint differs from specification")
        if str(cache.attrs["source_h5_sha256"]) != specification["bindings"]["source_h5_sha256"]:
            raise RuntimeError("cache source dataset differs from specification")
    payload = deepcopy(specification)
    payload["schema"] = "transfer_dg_parent_anchored_block32_training_preregistration_v1"
    payload["cache"] = {
        "path": str(args.cache.resolve()),
        "sha256": sha256(args.cache),
        "preregistration": str(args.cache_preregistration.resolve()),
        "preregistration_sha256": sha256(args.cache_preregistration),
    }
    payload["bindings"]["cache_sha256"] = payload["cache"]["sha256"]
    payload["bindings"]["cache_preregistration_sha256"] = payload["cache"][
        "preregistration_sha256"
    ]
    atomic_json(payload, args.output)
    print(json.dumps({"status": "complete", "output": str(args.output.resolve()), "sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
