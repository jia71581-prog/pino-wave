#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2

CACHE="results/r40_frequency_cache_v2_20260828"
OUTPUT="results/r40_frequency_residual_fno_w48_v1_20260828"
PYTHON="/root/miniconda3/bin/python"
TORCHRUN="/root/miniconda3/bin/torchrun"

if [[ ! -f "${CACHE}/bundle_summary.json" ]]; then
  echo "R40 v2 cache bundle is not complete" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite existing ${OUTPUT}" >&2
  exit 2
fi

"${PYTHON}" - "${CACHE}/bundle_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema") != "r40_frequency_cache_bundle_v2":
    raise RuntimeError("unexpected R40 cache bundle schema")
if payload.get("status") != "complete":
    raise RuntimeError("R40 cache bundle is not complete")
if payload.get("fit_record_count") != 1032:
    raise RuntimeError("unexpected R40 fit record count")
if payload.get("holdout_record_count") != 56:
    raise RuntimeError("unexpected R40 holdout record count")
if payload.get("validation_opened") or payload.get("test_id_opened"):
    raise RuntimeError("R40 cache evidence boundary violation")
PY

PYTHONPATH=scripts "${TORCHRUN}" --standalone --nproc-per-node=4 \
  scripts/train_r40_frequency_residual_operator.py \
  --fit-cache \
    "${CACHE}/fit_shard0.h5" \
    "${CACHE}/fit_shard1.h5" \
    "${CACHE}/fit_shard2.h5" \
    "${CACHE}/fit_shard3.h5" \
  --holdout-cache \
    "${CACHE}/holdout_shard0.h5" \
    "${CACHE}/holdout_shard1.h5" \
    "${CACHE}/holdout_shard2.h5" \
    "${CACHE}/holdout_shard3.h5" \
  --output-dir "${OUTPUT}" \
  --epochs 50 \
  --max-steps-per-epoch 0 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --eval-every 2 \
  --learning-rate 3e-4 \
  --weight-decay 1e-5 \
  --warmup-epochs 1 \
  --width 48 \
  --modes 16 \
  --blocks 4 \
  --correction-cap 1.5 \
  --tail-weight 1.0 \
  --hinge-weight 2.0 \
  --shape-weight 0.002 \
  --seed 400828 \
  --amp \
  --stop-on-pass
