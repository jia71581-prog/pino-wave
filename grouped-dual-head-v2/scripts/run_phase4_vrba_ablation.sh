#!/bin/bash
# Phase 4 vRBA ablation: 3 sequential 4-GPU DDP runs from the SOAP-best N420 checkpoint,
# same protocol as the SOAP-only baseline (records_per_family=420, 40ep, 32 frames,
# sigma2, helmholtz rank8/freq64). Each run ~18-20h; total ~55h. Logs to $ROOT_LOG.
#
# Runs (single-variable where possible):
#   1. per-frame baseline: --per-frame-frame, NO sampling. Isolates the loss-objective
#      change (per-frame-normalized vs record-normalized) so the frame-RBA line has a
#      clean control. Directly comparable to SOAP-only 0.0407 only up to this change.
#   2. rad-only: record RAD, per_frame_frame=False -> DIRECTLY comparable to SOAP-only
#      0.0407 (single variable = record sampling). Tests "does oversampling high-error
#      records help layered/marmousi undertrained components".
#   3. vrba: record RAD + frame RBA (needs --per-frame-frame). Compared against run 1.
#
# Verdict: beat SOAP-only 0.0407 aggregate and especially layered late 0.169. Honest
# boundary: late is a structural bottleneck (render rank collapse to 8, study S26-33);
# if sampling cannot move it, that is a valuable negative result.
set -u
cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
export HDF5_USE_FILE_LOCKING=FALSE
PY=/root/miniconda3/bin/python

NORM=/root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json
MERGED=/dev/shm/g3cache/background_pbg_sigma2_g3pool_N420.h5
# SOAP-best N420 checkpoint (held-out 0.0407) as the continue-pretraining start point.
CKPT=results/helmholtz_g3_aplus1_r8_sigma2_N420_ddp4_soap_frames32_ep40/checkpoints/update_1968.pt
ROOT_LOG=results/phase4_vrba_ablation.log

# Shared protocol flags (match the SOAP-only N420 run exactly).
COMMON="--warmstart-helmholtz $CKPT --background-cache $MERGED --normalization-json $NORM \
  --records-per-family 420 --helmholtz-rank 8 --helmholtz-frequencies 64 --optimizer soap \
  --dense-learning-rate 1e-4 --backbone-learning-rate 5e-5 --local-field-learning-rate 5e-4 \
  --epochs 40 --macro-records 6 --macros-per-update 4 --evaluate-every 48 --training-frames 32"

echo "[phase4] started $(date)" > $ROOT_LOG
echo "[phase4] start ckpt=$CKPT" >> $ROOT_LOG

run_one () {
  local name="$1"; shift
  local out="results/$name"
  mkdir -p "$out"
  if [ -f "$out/terminal.json" ]; then
    echo "[phase4] SKIP $name (terminal.json exists) $(date)" >> $ROOT_LOG
    return 0
  fi
  echo "[phase4] launching $name $(date)" >> $ROOT_LOG
  echo "[phase4]   extra flags: $*" >> $ROOT_LOG
  torchrun --nproc_per_node=4 --master_port=29561 scripts/diagnose_helmholtz_g3_heldout.py \
    $COMMON "$@" --artifact-dir "$out" >> "$out/train.stdout.log" 2>&1
  local rc=$?
  echo "[phase4] $name exited rc=$rc $(date)" >> $ROOT_LOG
  if [ $rc -ne 0 ]; then
    echo "[phase4] ABORT chain: $name failed" >> $ROOT_LOG
    exit $rc
  fi
}

# 2 is placed first: it is the only run directly comparable to the 0.0407 baseline
# (no loss-objective change), so it delivers the cleanest single-variable verdict first.
run_one helmholtz_g3_aplus1_soap_radonly_N420_ep40 \
  --adaptive-sampling rad --rad-recompute-epochs 5 --record-oversample 2.0 \
  --rad-potential quadratic --rad-uniform-fraction 0.2 --rad-ema-momentum 0.3

run_one helmholtz_g3_aplus1_soap_perframe_baseline_N420_ep40 \
  --per-frame-frame

run_one helmholtz_g3_aplus1_soap_vrba_N420_ep40 \
  --per-frame-frame \
  --adaptive-sampling vrba --rad-recompute-epochs 5 --record-oversample 2.0 \
  --rad-potential quadratic --rad-uniform-fraction 0.2 --rad-ema-momentum 0.3 \
  --frame-rba on --rba-potential quadratic --rba-gamma 0.999 --rba-eta 0.01 --rba-phi 0.9

echo "[phase4] all runs complete $(date)" >> $ROOT_LOG
