#!/bin/bash
# Build four missing-record sparse P_bg shards, validate their union, then start DDP.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
TIME_POOL_COUNT="${FULL3203_TIME_POOL_COUNT:-128}"
export FULL3203_TIME_POOL_COUNT="$TIME_POOL_COUNT"
CACHE_DIR="${FULL3203_CACHE_DIR:-/dev/shm/g3cache/full3203_time${TIME_POOL_COUNT}}"
PRIMARY_CACHE="${FULL3203_PRIMARY_CACHE:-/dev/shm/g3cache/background_pbg_sigma2_g3pool_N420.h5}"
SOURCE_H5=/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5
FROZEN_CONFIG=/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/frozen_config.yaml

cd "$REPO_ROOT"
mkdir -p "$CACHE_DIR"
available_kib=$(df --output=avail /dev/shm | tail -1)
if [ "$available_kib" -lt 36700160 ]; then
  echo "Need at least 35 GiB free in /dev/shm before sparse-cache build" >&2
  exit 70
fi

pids=()
for shard in 0 1 2 3; do
  output="$CACHE_DIR/background_pbg_sigma2_missing_time${TIME_POOL_COUNT}_shard${shard}.h5"
  log="$CACHE_DIR/build_shard${shard}.log"
  if [ -e "$output" ]; then
    /root/miniconda3/bin/python - "$output" <<'PY'
import sys
import h5py
with h5py.File(sys.argv[1], "r", swmr=True) as handle:
    if str(handle.attrs.get("status", "")) != "complete":
        raise SystemExit(f"existing cache is incomplete; preserve and inspect it: {sys.argv[1]}")
PY
    echo "Reusing complete shard: $output"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$shard" HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1 \
    /root/miniconda3/bin/python scripts/build_smoothed_background_cache.py \
      --all-non-anomaly \
      --exclude-cache "$PRIMARY_CACHE" \
      --shard-count 4 \
      --shard-index "$shard" \
      --time-count "$TIME_POOL_COUNT" \
      --sigma 2.0 \
      --source-h5 "$SOURCE_H5" \
      --frozen-config "$FROZEN_CONFIG" \
      --device cuda \
      --out "$output" >"$log" 2>&1 &
  pids+=("$!")
  echo "Started background shard $shard pid=$! log=$log"
done

terminate_children() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [ "$failed" -ne 0 ]; then
  echo "At least one background shard failed; training was not started" >&2
  exit 1
fi

echo "All sparse background shards completed; starting audited full3203 continuation"
exec bash scripts/run_helmholtz_full3203_time96_continue_ddp4.sh
