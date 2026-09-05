#!/usr/bin/env bash
# Reflection-recovery continuation: train a global Born scattering propagator on
# p - P_bg and select checkpoints by scattering-field error.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
ARTIFACT_DIR="${REFLECTION_RECOVERY_ARTIFACT_DIR:?Set REFLECTION_RECOVERY_ARTIFACT_DIR}"
PARENT_TERMINAL="${REFLECTION_RECOVERY_PARENT_TERMINAL:-$REPO_ROOT/results/helmholtz_g3_phase4b_vrba_frames32_pf420_ep5_20260807T031404Z/terminal.json}"
SOURCE_H5="${REFLECTION_RECOVERY_SOURCE_H5:-/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5}"
NORMALIZATION_JSON="${REFLECTION_RECOVERY_NORMALIZATION_JSON:-/root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json}"
BACKGROUND_CACHE="${REFLECTION_RECOVERY_BACKGROUND_CACHE:-/dev/shm/g3cache/background_pbg_sigma2_g3pool_N420.h5}"
GPU_IDS="${REFLECTION_RECOVERY_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${REFLECTION_RECOVERY_NPROC_PER_NODE:-4}"
MACROS_PER_UPDATE="${REFLECTION_RECOVERY_MACROS_PER_UPDATE:-$NPROC}"
EPOCHS="${REFLECTION_RECOVERY_EPOCHS:-5}"
TRAINING_FRAMES="${REFLECTION_RECOVERY_TRAINING_FRAMES:-64}"
SMOKE_UPDATES="${REFLECTION_RECOVERY_SMOKE_UPDATES:-0}"
EVALUATE_EVERY="${REFLECTION_RECOVERY_EVALUATE_EVERY:-53}"
LOCAL_FIELD_LR="${REFLECTION_RECOVERY_LOCAL_FIELD_LR:-0.0005}"

cd "$REPO_ROOT"
if [ ! -s "$PARENT_TERMINAL" ]; then
  echo "Reflection-recovery parent terminal is missing: $PARENT_TERMINAL" >&2
  exit 66
fi
if [ ! -s "$SOURCE_H5" ]; then
  echo "Reflection-recovery source dataset is missing: $SOURCE_H5" >&2
  echo "Set REFLECTION_RECOVERY_SOURCE_H5 to the validated replacement/hybrid VDS." >&2
  exit 66
fi
if [ ! -s "$NORMALIZATION_JSON" ]; then
  echo "Reflection-recovery normalization is missing: $NORMALIZATION_JSON" >&2
  exit 66
fi
if [ ! -s "$BACKGROUND_CACHE" ]; then
  echo "Reflection-recovery P_bg cache is missing: $BACKGROUND_CACHE" >&2
  exit 66
fi
if [ -e "$ARTIFACT_DIR/run_identity.json" ] || [ -e "$ARTIFACT_DIR/terminal.json" ]; then
  echo "Refusing to overwrite an existing reflection-recovery run: $ARTIFACT_DIR" >&2
  exit 64
fi
mkdir -p "$ARTIFACT_DIR"

SMOKE_ARGS=()
if [ "$SMOKE_UPDATES" -gt 0 ]; then
  SMOKE_ARGS+=(--smoke-updates "$SMOKE_UPDATES")
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1

exec /root/miniconda3/bin/torchrun --standalone --nproc_per_node="$NPROC" \
  scripts/diagnose_helmholtz_g3_heldout.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --source-h5 "$SOURCE_H5" \
  --normalization-json "$NORMALIZATION_JSON" \
  --warmstart-best-terminal "$PARENT_TERMINAL" \
  --require-best-warmstart \
  --width 128 \
  --dense-depth 8 \
  --dense-spectral-rank 112 \
  --dense-modes 32 \
  --helmholtz-frequencies 64 \
  --helmholtz-rank 8 \
  --helmholtz-background-conditioning \
  --helmholtz-background-global-propagator \
  --helmholtz-background-propagation-modes 48 \
  --background-conditioner-only \
  --checkpoint-selection-metric scattering_relative_l2 \
  --optimizer adamw \
  --epochs "$EPOCHS" \
  --macro-records 6 \
  --macros-per-update "$MACROS_PER_UPDATE" \
  --microbatch-records 1 \
  --training-frames "$TRAINING_FRAMES" \
  --validation-frames 32 \
  --per-frame-frame \
  --frame-energy-floor-fraction 0 \
  --uniform-gradient-weight 0.1 \
  --layered-gradient-weight 1.0 \
  --marmousi-gradient-weight 1.0 \
  --local-field-learning-rate "$LOCAL_FIELD_LR" \
  --dense-learning-rate 0.00002 \
  --backbone-learning-rate 0.00001 \
  --evaluate-every "$EVALUATE_EVERY" \
  --travel-time-h5 /root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5 \
  --seed 372 \
  --background-cache "$BACKGROUND_CACHE" \
  --records-per-family 420 \
  --workers 2 \
  --prefetch-factor 2 \
  --adaptive-sampling vrba \
  --rad-recompute-epochs 1 \
  --record-oversample 2.0 \
  --rad-potential quadratic \
  --rad-uniform-fraction 0.2 \
  --rad-ema-momentum 0.3 \
  --frame-rba on \
  --rba-gamma 0.999 \
  --rba-eta 0.01 \
  --rba-phi 0.9 \
  --rba-potential quadratic \
  "${SMOKE_ARGS[@]}"
