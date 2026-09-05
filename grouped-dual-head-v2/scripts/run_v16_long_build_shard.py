#!/usr/bin/env python3
"""Build one shard of the v16 long-stage bundles and park it in /dev/shm.

Per-record independence holds in build_bundles (each iteration touches only its
own sample_id; parent inference is stateless), so sharding the id list across
workers reproduces the monolithic build.  The frozen runner is imported, never
modified."""
import importlib.util
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "train_r16_dscp_v16", ROOT / "scripts/train_r16_dscp_v16.py"
)
v16 = importlib.util.module_from_spec(spec)
sys.modules["train_r16_dscp_v16"] = v16
spec.loader.exec_module(v16)


def main() -> int:
    shard, num_shards = int(sys.argv[1]), int(sys.argv[2])
    out_dir = Path("/dev/shm/v16_long_bundles")
    out_dir.mkdir(parents=True, exist_ok=True)
    fit = v16.role_sample_ids(v16.STAGE_PLAN["long"]["fit_role"])
    calib = v16.role_sample_ids(v16.STAGE_PLAN["long"]["eval_role"])
    all_ids = list(dict.fromkeys([*fit, *calib]))
    my_ids = all_ids[shard::num_shards]
    log = lambda m: print(f"{v16.utc_now()} [shard {shard}/{num_shards}] {m}", flush=True)
    log(f"{len(my_ids)} of {len(all_ids)} bundles")
    started = time.monotonic()
    bundles, context = v16.build_bundles(torch.device("cuda:0"), my_ids, log=log)
    payload = {
        "shard": shard, "num_shards": num_shards, "sample_ids": my_ids,
        "bundles": bundles, "bases": context["bases"], "scales": context["scales"],
        "build_s": time.monotonic() - started,
    }
    temp = out_dir / f".shard_{shard}.pt.tmp"
    torch.save(payload, temp)
    temp.rename(out_dir / f"shard_{shard}.pt")
    log(f"done in {payload['build_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
