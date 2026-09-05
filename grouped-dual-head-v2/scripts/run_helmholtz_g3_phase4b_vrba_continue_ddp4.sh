#!/bin/bash
# Four-GPU Phase4b continuation with record RAD + frame RBA residual sampling.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
ARTIFACT_DIR="${PHASE4B_VRBA_ARTIFACT_DIR:?Set PHASE4B_VRBA_ARTIFACT_DIR}"
PARENT_CHECKPOINT="${PHASE4B_VRBA_PARENT_CHECKPOINT:-$REPO_ROOT/results/archived_best/phase4b_perframe_baseline_before_chonknoris_20260807/best_aggregate_update_1824.pt}"
EPOCHS="${PHASE4B_VRBA_EPOCHS:-5}"
TRAINING_FRAMES="${PHASE4B_VRBA_TRAINING_FRAMES:-64}"
SMOKE_UPDATES="${PHASE4B_VRBA_SMOKE_UPDATES:-0}"
EVALUATE_EVERY="${PHASE4B_VRBA_EVALUATE_EVERY:-53}"
LOCAL_FIELD_LR="${PHASE4B_VRBA_LOCAL_FIELD_LR:-0.0001}"
DENSE_LR="${PHASE4B_VRBA_DENSE_LR:-0.00002}"
BACKBONE_LR="${PHASE4B_VRBA_BACKBONE_LR:-0.00001}"
RECORD_OVERSAMPLE="${PHASE4B_VRBA_RECORD_OVERSAMPLE:-2.0}"
RAD_UNIFORM_FRACTION="${PHASE4B_VRBA_UNIFORM_FRACTION:-0.2}"
UNIFORM_GRADIENT_WEIGHT="${PHASE4B_VRBA_UNIFORM_GRADIENT_WEIGHT:-0.1}"

cd "$REPO_ROOT"
if [ ! -s "$PARENT_CHECKPOINT" ]; then
  echo "Phase4b parent checkpoint is missing: $PARENT_CHECKPOINT" >&2
  exit 66
fi
if [ -e "$ARTIFACT_DIR/run_identity.json" ] || [ -e "$ARTIFACT_DIR/terminal.json" ]; then
  echo "Refusing to overwrite an existing vRBA run: $ARTIFACT_DIR" >&2
  exit 64
fi
mkdir -p "$ARTIFACT_DIR"

SMOKE_ARGS=()
if [ "$SMOKE_UPDATES" -gt 0 ]; then
  SMOKE_ARGS+=(--smoke-updates "$SMOKE_UPDATES")
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1

exec /root/miniconda3/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/diagnose_helmholtz_g3_heldout.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --normalization-json /root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json \
  --warmstart-helmholtz "$PARENT_CHECKPOINT" \
  --width 128 \
  --dense-depth 8 \
  --dense-spectral-rank 112 \
  --dense-modes 32 \
  --helmholtz-frequencies 64 \
  --helmholtz-rank 8 \
  --optimizer soap \
  --epochs "$EPOCHS" \
  --macro-records 6 \
  --macros-per-update 4 \
  --microbatch-records 1 \
  --training-frames "$TRAINING_FRAMES" \
  --validation-frames 32 \
  --per-frame-frame \
  --frame-energy-floor-fraction 0 \
  --uniform-gradient-weight "$UNIFORM_GRADIENT_WEIGHT" \
  --layered-gradient-weight 1.0 \
  --marmousi-gradient-weight 1.0 \
  --local-field-learning-rate "$LOCAL_FIELD_LR" \
  --dense-learning-rate "$DENSE_LR" \
  --backbone-learning-rate "$BACKBONE_LR" \
  --evaluate-every "$EVALUATE_EVERY" \
  --travel-time-h5 /root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5 \
  --seed 372 \
  --background-cache /dev/shm/g3cache/background_pbg_sigma2_g3pool_N420.h5 \
  --records-per-family 420 \
  --workers 2 \
  --prefetch-factor 2 \
  --adaptive-sampling vrba \
  --rad-recompute-epochs 1 \
  --record-oversample "$RECORD_OVERSAMPLE" \
  --rad-potential quadratic \
  --rad-uniform-fraction "$RAD_UNIFORM_FRACTION" \
  --rad-ema-momentum 0.3 \
  --frame-rba on \
  --rba-gamma 0.999 \
  --rba-eta 0.01 \
  --rba-phi 0.9 \
  --rba-potential quadratic \
  "${SMOKE_ARGS[@]}"
