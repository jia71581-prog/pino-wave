#!/usr/bin/env bash
# Wait for the detached Marmousi P_bg builders, validate their exact coverage, and
# only then hand control to the audited mixed-v5 four-GPU training launcher.
set -Eeuo pipefail

: "${ORIGINAL_REFLECTION_ARTIFACT_DIR:?Set ORIGINAL_REFLECTION_ARTIFACT_DIR}"
: "${ORIGINAL_REFLECTION_PARENT_CHECKPOINT:?Set ORIGINAL_REFLECTION_PARENT_CHECKPOINT}"

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
MIXED_ROOT=/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_train_marmousi_hires_v5_dt1e5_t401_mixed
CACHE_ROOT="$MIXED_ROOT/background_pbg_sigma2_marm700_full401"

while pgrep -f '[b]uild_smoothed_background_cache.py.*background_pbg_sigma2_marm700_full401' >/dev/null; do
  sleep 30
done

/root/miniconda3/bin/python - "$MIXED_ROOT/dataset_v5_dt1e5_t401_mixed.h5" "$CACHE_ROOT" <<'PY'
from pathlib import Path
import sys

import h5py
import numpy as np

source_path = Path(sys.argv[1]).resolve()
cache_root = Path(sys.argv[2]).resolve()
paths = tuple(cache_root / f"shard-{index}.h5" for index in range(4))
if any(not path.is_file() for path in paths):
    raise SystemExit("mixed background cache is missing one or more shards")

with h5py.File(source_path, "r", swmr=True) as source:
    expected_manifest = str(source.attrs.get("manifest_sha256", ""))

sample_ids: list[str] = []
source_indices: list[int] = []
for shard_index, path in enumerate(paths):
    with h5py.File(path, "r", swmr=True) as cache:
        if str(cache.attrs.get("status", "")) != "complete":
            raise SystemExit(f"background shard is not complete: {path}")
        if int(cache.attrs.get("shard_count", -1)) != 4 or int(
            cache.attrs.get("shard_index", -1)
        ) != shard_index:
            raise SystemExit(f"background shard identity mismatch: {path}")
        if str(cache.attrs.get("source_manifest_sha256", "")) != expected_manifest:
            raise SystemExit(f"background shard source manifest mismatch: {path}")
        indices = np.asarray(cache["source_index"][:], dtype=np.int64)
        times = np.asarray(cache["time_indices"][:], dtype=np.int64)
        wavefield = cache["wavefield"]
        if wavefield.shape != (len(indices), 401, 201, 201):
            raise SystemExit(f"background shard wavefield shape mismatch: {path}")
        if not np.array_equal(times, np.arange(401, dtype=np.int64)):
            raise SystemExit(f"background shard time coverage mismatch: {path}")
        source_indices.extend(int(value) for value in indices)
        sample_ids.extend(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in cache["sample_id"][:]
        )

if sorted(source_indices) != list(range(2100, 2800)):
    raise SystemExit("mixed background cache does not exactly cover source indices 2100..2799")
if len(sample_ids) != 700 or len(set(sample_ids)) != 700:
    raise SystemExit("mixed background cache sample IDs are not 700 unique records")
print("mixed background cache gate PASS: 4 shards, 700 records, 401x201x201", flush=True)
PY

exec "$REPO_ROOT/scripts/run_helmholtz_reflection_recovery_mixed_stage2_ddp4.sh"
