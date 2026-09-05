#!/bin/bash
# Reproducible detached launcher for train-only r5b CPADC basis fitting.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
CONFIG_PATH=configs/saved_time_v5/causal_defect_basis_r5b_marmousi1_4m_v2.yaml
RUN_DIR="${CPADC_RUN_DIR:?CPADC_RUN_DIR is required}"
VISIBLE_GPUS="${CPADC_VISIBLE_GPUS:-0}"
WORLD_SIZE="${CPADC_WORLD_SIZE:-1}"
PER_FAMILY="${CPADC_PER_FAMILY:-1}"
EPOCHS="${CPADC_EPOCHS:-3}"
RANK="${CPADC_RANK:-8}"
PHASE_RANK="${CPADC_PHASE_RANK:-2}"
WIDTH="${CPADC_WIDTH:-16}"
LEARNING_RATE="${CPADC_LEARNING_RATE:-5e-6}"
WARMUP_EPOCHS="${CPADC_WARMUP_EPOCHS:-1}"
PHYSICS_POINTS="${CPADC_PHYSICS_POINTS:-64}"
RANK_CHUNK_SIZE="${CPADC_RANK_CHUNK_SIZE:-2}"

if [ -e "$RUN_DIR/best.pt" ] || [ -e "$RUN_DIR/latest.pt" ] || [ -e "$RUN_DIR/terminal.json" ]; then
  echo "Refusing to overwrite a completed or partial CPADC run: $RUN_DIR" >&2
  exit 64
fi

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:src

exec /root/miniconda3/bin/torchrun \
  --standalone \
  --nproc_per_node "$WORLD_SIZE" \
  scripts/train_causal_defect_basis.py \
  --config "$CONFIG_PATH" \
  --output-dir "$RUN_DIR" \
  --device cuda \
  --per-family "$PER_FAMILY" \
  --epochs "$EPOCHS" \
  --patch-size 201 \
  --rank "$RANK" \
  --phase-rank "$PHASE_RANK" \
  --width "$WIDTH" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --physics-point-count "$PHYSICS_POINTS" \
  --rank-chunk-size "$RANK_CHUNK_SIZE"
