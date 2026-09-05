#!/usr/bin/env python3
"""Run the v16 long stage with bundles preloaded from /dev/shm shards.

Monkeypatches only the module attribute build_bundles; run_stage resolves it at
call time, so training semantics (seed, order, gates, checkpoints) are exactly
the frozen runner's.  Cold start: standalone long passes input_checkpoint=None."""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "train_r16_dscp_v16", ROOT / "scripts/train_r16_dscp_v16.py"
)
v16 = importlib.util.module_from_spec(spec)
sys.modules["train_r16_dscp_v16"] = v16
spec.loader.exec_module(v16)

SHARD_DIR = Path("/dev/shm/v16_long_bundles")


def build_bundles_from_shards(device, sample_ids, *, log):
    by_id, bases, scales, total_s = {}, None, None, 0.0
    shards = sorted(SHARD_DIR.glob("shard_*.pt"))
    if not shards:
        raise v16.V16Refusal("no prebuilt shards found")
    for path in shards:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for bundle in payload["bundles"]:
            by_id[bundle.sample_id] = bundle
        bases, scales = payload["bases"], payload["scales"]
        total_s += payload["build_s"]
    missing = [s for s in sample_ids if s not in by_id]
    if missing:
        raise v16.V16Refusal(f"prebuilt shards missing records: {missing[:5]}")
    log(f"loaded {len(by_id)} prebuilt bundles from {len(shards)} shards "
        f"(aggregate build {total_s:.0f}s)")
    bundles = [by_id[s] for s in sample_ids]
    context = {"bases": bases, "scales": scales, "device": device,
               "cache_build_s": total_s}
    return bundles, context


v16.build_bundles = build_bundles_from_shards

if __name__ == "__main__":
    raise SystemExit(v16.main(["long"]))
