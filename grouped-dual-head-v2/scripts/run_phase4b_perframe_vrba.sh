#!/bin/bash
# Phase 4b: resume the vRBA ablation chain AFTER rad-only was interrupted as a confirmed
# negative result (see results/helmholtz_g3_aplus1_soap_radonly_N420_ep40/NEGATIVE_RESULT.json:
# record-level RAD froze layered/marmousi from warmstart across all 49 evals; aggregate
# microdrop 0.0409->0.0407 was entirely the trivial uniform family). rad-only is therefore
# SKIPPED here to free the 4 GPUs for the two runs that actually probe the late-time
# structural bottleneck via frame-level RBA:
#
#   1. perframe_baseline: --per-frame-frame, NO sampling. Clean single-variable control
#      for the frame-RBA line (isolates the per-frame-normalized loss objective).
#   2. vrba: record RAD + frame RBA (needs --per-frame-frame). Compared against run 1 to
#      isolate whether bounded-EMA frame attention moves layered late (0.169) at all.
#
# Verdict: beat perframe_baseline on layered late. Honest boundary: late is a structural
# bottleneck (render rank collapse to 8, study S26-33); if frame RBA cannot move it, that
# is a valuable negative result and closes the sampling line.
set -u
cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
export HDF5_USE_FILE_LOCKING=FALSE
PY=/root/miniconda3/bin/python

NORM=/root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json
MERGED=/dev/shm/g3cache/background_pbg_sigma2_g3pool_N420.h5
CKPT=results/helmholtz_g3_aplus1_r8_sigma2_N420_ddp4_soap_frames32_ep40/checkpoints/update_1968.pt
ROOT_LOG=results/phase4b_perframe_vrba.log

COMMON="--warmstart-helmholtz $CKPT --background-cache $MERGED --normalization-json $NORM \
  --records-per-family 420 --helmholtz-rank 8 --helmholtz-frequencies 64 --optimizer soap \
  --dense-learning-rate 1e-4 --backbone-learning-rate 5e-5 --local-field-learning-rate 5e-4 \
  --epochs 40 --macro-records 6 --macros-per-update 4 --evaluate-every 48 --training-frames 32"

echo "[phase4b] started $(date)" > $ROOT_LOG
echo "[phase4b] start ckpt=$CKPT (rad-only SKIPPED: confirmed negative)" >> $ROOT_LOG

run_one () {
  local name="$1"; shift
  local out="results/$name"
  mkdir -p "$out"
  if [ -f "$out/terminal.json" ]; then
    echo "[phase4b] SKIP $name (terminal.json exists) $(date)" >> $ROOT_LOG
    return 0
  fi
  echo "[phase4b] launching $name $(date)" >> $ROOT_LOG
  echo "[phase4b]   extra flags: $*" >> $ROOT_LOG
  torchrun --nproc_per_node=4 --master_port=29563 scripts/diagnose_helmholtz_g3_heldout.py \
    $COMMON "$@" --artifact-dir "$out" >> "$out/train.stdout.log" 2>&1
  local rc=$?
  echo "[phase4b] $name exited rc=$rc $(date)" >> $ROOT_LOG
  if [ $rc -ne 0 ]; then
    echo "[phase4b] ABORT chain: $name failed" >> $ROOT_LOG
    exit $rc
  fi
}

run_one helmholtz_g3_aplus1_soap_perframe_baseline_N420_ep40 \
  --per-frame-frame

run_one helmholtz_g3_aplus1_soap_vrba_N420_ep40 \
  --per-frame-frame \
  --adaptive-sampling vrba --rad-recompute-epochs 5 --record-oversample 2.0 \
  --rad-potential quadratic --rad-uniform-fraction 0.2 --rad-ema-momentum 0.3 \
  --frame-rba on --rba-potential quadratic --rba-gamma 0.999 --rba-eta 0.01 --rba-phi 0.9

echo "[phase4b] all runs complete $(date)" >> $ROOT_LOG
