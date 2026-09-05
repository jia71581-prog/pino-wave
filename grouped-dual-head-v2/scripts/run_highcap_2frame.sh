#!/usr/bin/env bash
# 2-frame high-capacity per-instance fine-tune ceiling (fixed obs-loss norm).
# Reads ONLY the two early onset frames (causal); full residual head unfrozen
# + PDE self-supervision on unobserved times. Reports held-out relL2 ceiling.
set -euo pipefail
WORKROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$WORKROOT"
CFG=configs/saved_time_v5/meta_hypernet.yaml
OUT=results/instance_adaptation/highcap_2frame
mkdir -p "$OUT"
GPU="${1:-0}"; shift || true
for SID in "$@"; do
  LOG="$OUT/${SID}.json"
  [ -s "$LOG" ] && { echo "skip $LOG"; continue; }
  echo "=== $SID (gpu $GPU) ==="
  PYTHONPATH="$WORKROOT" CUDA_VISIBLE_DEVICES="$GPU" \
    python scripts/diagnose_highcap_instance_finetune.py \
      --config "$CFG" --sample-id "$SID" --steps 50,150,400,800 --pde-weight 0.1 \
    > "$LOG.tmp" 2>&1 && mv "$LOG.tmp" "$LOG" || { echo "FAIL $SID"; mv "$LOG.tmp" "$LOG.failed"; }
done
echo "DONE highcap gpu $GPU"
