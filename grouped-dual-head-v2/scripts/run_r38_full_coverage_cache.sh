#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2"
OUT="$ROOT/results/r38_full_coverage_cache_20260828"
MANIFEST="$ROOT/results/r38_full_coverage_manifest_20260828.json"
PY="/root/miniconda3/bin/python"

cd "$ROOT"
if [[ -e "$OUT" ]]; then
  echo "refusing to reuse existing R38 cache directory: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"

pids=()
for gpu in 0 1 2 3; do
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/build_r25_coarse_residual_cache.py cache \
      --manifest "$MANIFEST" \
      --subset fit \
      --shard-index "$gpu" \
      --shard-count 4 \
      --output "$OUT/fit_shard_${gpu}.h5" \
      --solver-batch-size 8 \
      --device cuda:0
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/build_r25_coarse_residual_cache.py cache \
      --manifest "$MANIFEST" \
      --subset holdout \
      --shard-index "$gpu" \
      --shard-count 4 \
      --output "$OUT/holdout_shard_${gpu}.h5" \
      --solver-batch-size 6 \
      --device cuda:0
  ) >"$OUT/worker_${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "R38 cache generation failed; inspect worker logs" >&2
  exit "$status"
fi
for gpu in 0 1 2 3; do
  test -s "$OUT/fit_shard_${gpu}.h5.summary.json"
  test -s "$OUT/holdout_shard_${gpu}.h5.summary.json"
done
echo "R38_CACHE_COMPLETE"
