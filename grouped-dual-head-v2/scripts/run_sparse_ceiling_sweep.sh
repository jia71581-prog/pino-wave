#!/usr/bin/env bash
# Information-content ceiling sweep: for each family, how low can a MAX-capacity
# free full-field correction push held-out relL2 as we increase the number of
# sparse observed frames (spread across the energetic span) + PDE self-supervision.
# A free tensor has no capacity limit, so this isolates information vs structure.
set -euo pipefail
WORKROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$WORKROOT"
CFG=configs/saved_time_v5/meta_hypernet.yaml
OUT=results/instance_adaptation/ceiling_sweep
mkdir -p "$OUT"
GPU="${1:-0}"
shift || true
SAMPLES=("$@")   # sample ids to run on this GPU

for SID in "${SAMPLES[@]}"; do
  for N in 6 12 24 48 100; do
    LOG="$OUT/${SID}_nobs${N}.json"
    if [ -s "$LOG" ]; then echo "skip $LOG (exists)"; continue; fi
    echo "=== $SID  n_obs=$N  (gpu $GPU) ==="
    PYTHONPATH="$WORKROOT" CUDA_VISIBLE_DEVICES="$GPU" \
      python scripts/diagnose_sparse_frame_ceiling.py \
        --config "$CFG" --sample-id "$SID" --n-obs "$N" \
        --steps 100,400,800,1500 --pde-weight 0.1 \
      > "$LOG.tmp" 2>&1 && mv "$LOG.tmp" "$LOG" || { echo "FAILED $SID n=$N"; mv "$LOG.tmp" "$LOG.failed"; }
  done
done
echo "DONE gpu $GPU"
