#!/usr/bin/env bash
# Four-GPU stage-1 reflection continuation on the readable subset of the retained original VDS.
# The invalidation report is mandatory; deleted/fill Marmousi rows never enter the manifest.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
ARTIFACT_DIR="${ORIGINAL_REFLECTION_ARTIFACT_DIR:?Set ORIGINAL_REFLECTION_ARTIFACT_DIR}"
PARENT_TERMINAL="${ORIGINAL_REFLECTION_PARENT_TERMINAL:-$REPO_ROOT/results/helmholtz_g3_phase4b_vrba_frames32_pf420_ep5_20260807T031404Z/terminal.json}"
SOURCE_H5="${ORIGINAL_REFLECTION_SOURCE_H5:-/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.invalidated_source_for_hybrid_v2.h5}"
INVALIDATION_REPORT="${ORIGINAL_REFLECTION_INVALIDATION_REPORT:-/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.INVALID_AFTER_MARMOUSI_CLEANUP.json}"
NORMALIZATION_JSON="${ORIGINAL_REFLECTION_NORMALIZATION_JSON:-/root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json}"
BACKGROUND_CACHE="${ORIGINAL_REFLECTION_BACKGROUND_CACHE:-/dev/shm/g3cache/background_pbg_sigma2_g3pool_N420.h5}"
TRAVEL_TIME_H5="${ORIGINAL_REFLECTION_TRAVEL_TIME_H5:-/root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5}"
EPOCHS="${ORIGINAL_REFLECTION_EPOCHS:-5}"
TRAINING_FRAMES="${ORIGINAL_REFLECTION_TRAINING_FRAMES:-64}"
EVALUATE_EVERY="${ORIGINAL_REFLECTION_EVALUATE_EVERY:-71}"
LOCAL_FIELD_LR="${ORIGINAL_REFLECTION_LOCAL_FIELD_LR:-0.00001}"
SMOKE_UPDATES="${ORIGINAL_REFLECTION_SMOKE_UPDATES:-0}"

cd "$REPO_ROOT"
for required in "$PARENT_TERMINAL" "$SOURCE_H5" "$INVALIDATION_REPORT" \
  "$NORMALIZATION_JSON" "$BACKGROUND_CACHE" "$TRAVEL_TIME_H5"; do
  if [ ! -s "$required" ]; then
    echo "Required original-reflection input is missing: $required" >&2
    exit 66
  fi
done
if [ -e "$ARTIFACT_DIR/run_identity.json" ] || [ -e "$ARTIFACT_DIR/terminal.json" ]; then
  echo "Refusing to overwrite an existing original-reflection run: $ARTIFACT_DIR" >&2
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
  --source-h5 "$SOURCE_H5" \
  --source-invalidation-report "$INVALIDATION_REPORT" \
  --allow-filtered-invalidated-source \
  --normalization-json "$NORMALIZATION_JSON" \
  --allow-parent-normalization-binding \
  --travel-time-h5 "$TRAVEL_TIME_H5" \
  --allow-travel-source-path-alias \
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
  --background-conditioner-output-only \
  --checkpoint-selection-metric scattering_relative_l2 \
  --optimizer adamw \
  --epochs "$EPOCHS" \
  --macro-records 6 \
  --macros-per-update 4 \
  --microbatch-records 1 \
  --training-frames "$TRAINING_FRAMES" \
  --validation-frames 32 \
  --per-frame-frame \
  --frame-energy-floor-fraction 0.05 \
  --uniform-gradient-weight 0.1 \
  --layered-gradient-weight 1.0 \
  --marmousi-gradient-weight 1.0 \
  --uniform-sampling-weight 0.1 \
  --layered-sampling-weight 1.0 \
  --marmousi-sampling-weight 32.0 \
  --local-field-learning-rate "$LOCAL_FIELD_LR" \
  --dense-learning-rate 0.00002 \
  --backbone-learning-rate 0.00001 \
  --evaluate-every "$EVALUATE_EVERY" \
  --seed 372 \
  --background-cache "$BACKGROUND_CACHE" \
  --uniform-records 420 \
  --layered-records 420 \
  --marmousi-records 4 \
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
