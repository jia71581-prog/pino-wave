#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2

output_dir=results/r46_frequency_cache_marmousi_supplement_v1_20260828
if [[ -e "${output_dir}" ]]; then
  echo "refusing existing output directory: ${output_dir}" >&2
  exit 1
fi
mkdir -p "${output_dir}"

pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${shard}" /root/miniconda3/bin/python \
    scripts/build_r40_frequency_cache.py \
    --manifest results/r46_marmousi_supplement_manifest_v1_20260828.json \
    --subset fit \
    --shard-index "${shard}" \
    --shard-count 4 \
    --output "${output_dir}/fit_supplement_shard${shard}.h5" \
    --device cuda:0 \
    --solver-batch-size 4 \
    --model-batch-size 16 \
    --amp \
    > "${output_dir}/fit_supplement_shard${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
