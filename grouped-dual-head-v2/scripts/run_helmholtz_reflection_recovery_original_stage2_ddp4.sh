#!/usr/bin/env bash
# Four-GPU stage-2 reflection continuation: keep the parent frozen, unfreeze the
# complete Born conditioner, and use a lower LR for its fresh propagation core.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
ARTIFACT_DIR="${ORIGINAL_REFLECTION_ARTIFACT_DIR:?Set ORIGINAL_REFLECTION_ARTIFACT_DIR}"
PARENT_CHECKPOINT="${ORIGINAL_REFLECTION_PARENT_CHECKPOINT:?Set ORIGINAL_REFLECTION_PARENT_CHECKPOINT}"
SOURCE_H5="${ORIGINAL_REFLECTION_SOURCE_H5:-/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.invalidated_source_for_hybrid_v2.h5}"
INVALIDATION_REPORT="${ORIGINAL_REFLECTION_INVALIDATION_REPORT:-/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.INVALID_AFTER_MARMOUSI_CLEANUP.json}"
SOURCE_REPLACEMENT_GATE="${ORIGINAL_REFLECTION_SOURCE_REPLACEMENT_GATE:-}"
NORMALIZATION_JSON="${ORIGINAL_REFLECTION_NORMALIZATION_JSON:-/root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json}"
BACKGROUND_CACHE="${ORIGINAL_REFLECTION_BACKGROUND_CACHE:-/dev/shm/g3cache/background_pbg_sigma2_g3pool_N420.h5}"
BACKGROUND_CACHE_SHARDS="${ORIGINAL_REFLECTION_BACKGROUND_CACHE_SHARDS:-}"
TRAVEL_TIME_H5="${ORIGINAL_REFLECTION_TRAVEL_TIME_H5:-/root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5}"
EPOCHS="${ORIGINAL_REFLECTION_EPOCHS:-5}"
MACRO_RECORDS="${ORIGINAL_REFLECTION_MACRO_RECORDS:-8}"
MACROS_PER_UPDATE="${ORIGINAL_REFLECTION_MACROS_PER_UPDATE:-4}"
MICROBATCH_RECORDS="${ORIGINAL_REFLECTION_MICROBATCH_RECORDS:-1}"
TRAINING_FRAMES="${ORIGINAL_REFLECTION_TRAINING_FRAMES:-96}"
VALIDATION_FRAMES="${ORIGINAL_REFLECTION_VALIDATION_FRAMES:-401}"
PER_FRAME_FRAME="${ORIGINAL_REFLECTION_PER_FRAME_FRAME:-1}"
FRAME_ENERGY_FLOOR="${ORIGINAL_REFLECTION_FRAME_ENERGY_FLOOR:-0.05}"
FULL_FIELD_FRAME_WEIGHT="${ORIGINAL_REFLECTION_FULL_FIELD_FRAME_WEIGHT:-0.0}"
DELTA_LOSS_WEIGHT="${ORIGINAL_REFLECTION_DELTA_LOSS_WEIGHT:-1.0}"
DELTA_REFERENCE="${ORIGINAL_REFLECTION_DELTA_REFERENCE:-target}"
DELTA_REDUCTION="${ORIGINAL_REFLECTION_DELTA_REDUCTION:-global_squared}"
DELTA_ENERGY_FLOOR="${ORIGINAL_REFLECTION_DELTA_ENERGY_FLOOR:-0.0}"
SPATIAL_GRADIENT_LOSS_WEIGHT="${ORIGINAL_REFLECTION_SPATIAL_GRADIENT_LOSS_WEIGHT:-0.0}"
EVALUATE_EVERY="${ORIGINAL_REFLECTION_EVALUATE_EVERY:-71}"
EVALUATION_SPLIT="${ORIGINAL_REFLECTION_EVALUATION_SPLIT:-validation}"
UNIFORM_RECORDS="${ORIGINAL_REFLECTION_UNIFORM_RECORDS:-420}"
LAYERED_RECORDS="${ORIGINAL_REFLECTION_LAYERED_RECORDS:-420}"
MARMOUSI_RECORDS="${ORIGINAL_REFLECTION_MARMOUSI_RECORDS:-4}"
UNIFORM_SAMPLING_WEIGHT="${ORIGINAL_REFLECTION_UNIFORM_SAMPLING_WEIGHT:-0.1}"
LAYERED_SAMPLING_WEIGHT="${ORIGINAL_REFLECTION_LAYERED_SAMPLING_WEIGHT:-1.0}"
MARMOUSI_SAMPLING_WEIGHT="${ORIGINAL_REFLECTION_MARMOUSI_SAMPLING_WEIGHT:-32.0}"
OUTPUT_LR="${ORIGINAL_REFLECTION_OUTPUT_LR:-0.0000001}"
CORE_LR="${ORIGINAL_REFLECTION_CORE_LR:-0.0000001}"
COUPLED_LR="${ORIGINAL_REFLECTION_COUPLED_LR:-}"
EXPERT_LR="${ORIGINAL_REFLECTION_EXPERT_LR:-}"
EXPERT_ROUTER_LR="${ORIGINAL_REFLECTION_EXPERT_ROUTER_LR:-}"
ADAM_EPSILON="${ORIGINAL_REFLECTION_ADAM_EPSILON:-0.00000001}"
LATE_RANK="${ORIGINAL_REFLECTION_LATE_RANK:-32}"
LATE_FREQUENCIES="${ORIGINAL_REFLECTION_LATE_FREQUENCIES:-24}"
LATE_LR="${ORIGINAL_REFLECTION_LATE_LR:-0.0000001}"
DIRECT_FREQUENCY_HEAD="${ORIGINAL_REFLECTION_DIRECT_FREQUENCY_HEAD:-0}"
DIRECT_FREQUENCIES="${ORIGINAL_REFLECTION_DIRECT_FREQUENCIES:-32}"
DIRECT_SPECTRAL_EXPERTS="${ORIGINAL_REFLECTION_DIRECT_SPECTRAL_EXPERTS:-0}"
FOURIER_UNIFORM_TIMES="${ORIGINAL_REFLECTION_FOURIER_UNIFORM_TIMES:-$DIRECT_FREQUENCY_HEAD}"
ADAPTIVE_SAMPLING="${ORIGINAL_REFLECTION_ADAPTIVE_SAMPLING:-vrba}"
RAD_RECOMPUTE_EPOCHS="${ORIGINAL_REFLECTION_RAD_RECOMPUTE_EPOCHS:-1}"
RECORD_OVERSAMPLE="${ORIGINAL_REFLECTION_RECORD_OVERSAMPLE:-2.0}"
FRAME_RBA="${ORIGINAL_REFLECTION_FRAME_RBA:-on}"
SMOKE_UPDATES="${ORIGINAL_REFLECTION_SMOKE_UPDATES:-0}"

cd "$REPO_ROOT"
required_inputs=(
  "$PARENT_CHECKPOINT" "$SOURCE_H5" "$NORMALIZATION_JSON"
  "$BACKGROUND_CACHE" "$TRAVEL_TIME_H5"
)
SOURCE_AUDIT_ARGS=()
if [ -n "$SOURCE_REPLACEMENT_GATE" ]; then
  required_inputs+=("$SOURCE_REPLACEMENT_GATE")
  SOURCE_AUDIT_ARGS+=(
    --source-replacement-gate "$SOURCE_REPLACEMENT_GATE"
    --allow-parent-normalization-binding
  )
else
  required_inputs+=("$INVALIDATION_REPORT")
  SOURCE_AUDIT_ARGS+=(
    --source-invalidation-report "$INVALIDATION_REPORT"
    --allow-filtered-invalidated-source
    --allow-parent-normalization-binding
    --allow-travel-source-path-alias
  )
fi

BACKGROUND_CACHE_ARGS=(--background-cache "$BACKGROUND_CACHE")
if [ -n "$BACKGROUND_CACHE_SHARDS" ]; then
  IFS=: read -r -a background_shards <<< "$BACKGROUND_CACHE_SHARDS"
  for shard in "${background_shards[@]}"; do
    [ -n "$shard" ] || continue
    required_inputs+=("$shard")
    BACKGROUND_CACHE_ARGS+=(--background-cache-shard "$shard")
  done
fi

for required in "${required_inputs[@]}"; do
  if [ ! -s "$required" ]; then
    echo "Required stage-2 reflection input is missing: $required" >&2
    exit 66
  fi
done
if [ -e "$ARTIFACT_DIR/run_identity.json" ] || [ -e "$ARTIFACT_DIR/terminal.json" ]; then
  echo "Refusing to overwrite an existing stage-2 reflection run: $ARTIFACT_DIR" >&2
  exit 64
fi
mkdir -p "$ARTIFACT_DIR"

SMOKE_ARGS=()
if [ "$SMOKE_UPDATES" -gt 0 ]; then
  SMOKE_ARGS+=(--smoke-updates "$SMOKE_UPDATES")
fi

REPRESENTATION_ARGS=()
if [ "$DIRECT_FREQUENCY_HEAD" = "1" ]; then
  REPRESENTATION_ARGS+=(
    --helmholtz-background-direct-frequency-head
    --helmholtz-background-direct-frequencies "$DIRECT_FREQUENCIES"
    --helmholtz-background-direct-spectral-experts "$DIRECT_SPECTRAL_EXPERTS"
  )
else
  REPRESENTATION_ARGS+=(
    --helmholtz-late-rank "$LATE_RANK"
    --helmholtz-late-frequencies "$LATE_FREQUENCIES"
    --background-conditioner-with-late-head
    --helmholtz-late-learning-rate "$LATE_LR"
  )
fi

TIME_POLICY_ARGS=()
if [ "$FOURIER_UNIFORM_TIMES" = "1" ]; then
  TIME_POLICY_ARGS+=(--fourier-uniform-training-times)
fi

COUPLED_LR_ARGS=()
if [ -n "$COUPLED_LR" ]; then
  COUPLED_LR_ARGS+=(
    --background-conditioner-coupled-learning-rate "$COUPLED_LR"
  )
fi

EXPERT_LR_ARGS=()
if [ -n "$EXPERT_LR" ]; then
  EXPERT_LR_ARGS+=(
    --background-conditioner-expert-learning-rate "$EXPERT_LR"
  )
fi

EXPERT_ROUTER_LR_ARGS=()
if [ -n "$EXPERT_ROUTER_LR" ]; then
  EXPERT_ROUTER_LR_ARGS+=(
    --background-conditioner-expert-router-learning-rate "$EXPERT_ROUTER_LR"
  )
fi

LOSS_ARGS=()
if [ "$PER_FRAME_FRAME" = "1" ]; then
  LOSS_ARGS+=(
    --per-frame-frame
    --frame-energy-floor-fraction "$FRAME_ENERGY_FLOOR"
  )
elif [ "$PER_FRAME_FRAME" != "0" ]; then
  echo "ORIGINAL_REFLECTION_PER_FRAME_FRAME must be 0 or 1" >&2
  exit 64
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
  "${SOURCE_AUDIT_ARGS[@]}" \
  --normalization-json "$NORMALIZATION_JSON" \
  --travel-time-h5 "$TRAVEL_TIME_H5" \
  --warmstart-helmholtz "$PARENT_CHECKPOINT" \
  --width 128 \
  --dense-depth 8 \
  --dense-spectral-rank 112 \
  --dense-modes 32 \
  --helmholtz-frequencies 64 \
  --helmholtz-rank 8 \
  --helmholtz-background-conditioning \
  --helmholtz-background-global-propagator \
  --helmholtz-background-propagation-modes 48 \
  "${REPRESENTATION_ARGS[@]}" \
  --background-conditioner-only \
  --background-conditioner-output-learning-rate "$OUTPUT_LR" \
  --background-conditioner-core-learning-rate "$CORE_LR" \
  "${COUPLED_LR_ARGS[@]}" \
  "${EXPERT_LR_ARGS[@]}" \
  "${EXPERT_ROUTER_LR_ARGS[@]}" \
  --background-conditioner-adam-epsilon "$ADAM_EPSILON" \
  --checkpoint-selection-metric scattering_relative_l2 \
  --optimizer adamw \
  --epochs "$EPOCHS" \
  --macro-records "$MACRO_RECORDS" \
  --macros-per-update "$MACROS_PER_UPDATE" \
  --microbatch-records "$MICROBATCH_RECORDS" \
  --training-frames "$TRAINING_FRAMES" \
  "${TIME_POLICY_ARGS[@]}" \
  --validation-frames "$VALIDATION_FRAMES" \
  --evaluation-split "$EVALUATION_SPLIT" \
  --full-field-frame-weight "$FULL_FIELD_FRAME_WEIGHT" \
  --delta-loss-weight "$DELTA_LOSS_WEIGHT" \
  --delta-reference "$DELTA_REFERENCE" \
  --delta-reduction "$DELTA_REDUCTION" \
  --delta-energy-floor-fraction "$DELTA_ENERGY_FLOOR" \
  --spatial-gradient-loss-weight "$SPATIAL_GRADIENT_LOSS_WEIGHT" \
  "${LOSS_ARGS[@]}" \
  --uniform-gradient-weight 0.1 \
  --layered-gradient-weight 1.0 \
  --marmousi-gradient-weight 1.0 \
  --uniform-sampling-weight "$UNIFORM_SAMPLING_WEIGHT" \
  --layered-sampling-weight "$LAYERED_SAMPLING_WEIGHT" \
  --marmousi-sampling-weight "$MARMOUSI_SAMPLING_WEIGHT" \
  --local-field-learning-rate "$CORE_LR" \
  --dense-learning-rate 0.00002 \
  --backbone-learning-rate 0.00001 \
  --evaluate-every "$EVALUATE_EVERY" \
  --seed 372 \
  "${BACKGROUND_CACHE_ARGS[@]}" \
  --uniform-records "$UNIFORM_RECORDS" \
  --layered-records "$LAYERED_RECORDS" \
  --marmousi-records "$MARMOUSI_RECORDS" \
  --workers 2 \
  --prefetch-factor 2 \
  --adaptive-sampling "$ADAPTIVE_SAMPLING" \
  --rad-recompute-epochs "$RAD_RECOMPUTE_EPOCHS" \
  --record-oversample "$RECORD_OVERSAMPLE" \
  --rad-potential quadratic \
  --rad-uniform-fraction 0.2 \
  --rad-ema-momentum 0.3 \
  --frame-rba "$FRAME_RBA" \
  --rba-gamma 0.999 \
  --rba-eta 0.01 \
  --rba-phi 0.9 \
  --rba-potential quadratic \
  "${SMOKE_ARGS[@]}"
