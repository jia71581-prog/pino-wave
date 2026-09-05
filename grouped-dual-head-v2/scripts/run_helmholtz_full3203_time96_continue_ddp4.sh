#!/bin/bash
# Four-GPU continuation over every non-anomaly sample and a dense exact-time pool.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
ARTIFACT_DIR="${FULL3203_ARTIFACT_DIR:?Set FULL3203_ARTIFACT_DIR}"
PARENT_TERMINAL="${FULL3203_PARENT_TERMINAL:-}"
SEED_TERMINAL=$REPO_ROOT/results/helmholtz_g3_phase4b_vrba_frames32_pf420_ep5_20260807T031404Z/terminal.json
PRIMARY_CACHE="${FULL3203_PRIMARY_CACHE:-/dev/shm/g3cache/background_pbg_sigma2_g3pool_N420.h5}"
TIME_POOL_COUNT="${FULL3203_TIME_POOL_COUNT:-128}"
CACHE_DIR="${FULL3203_CACHE_DIR:-/dev/shm/g3cache/full3203_time${TIME_POOL_COUNT}}"
TRAVEL_CACHE="${FULL3203_TRAVEL_CACHE:-/root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_all_nonanomaly_v2.h5}"
EPOCHS="${FULL3203_EPOCHS:-5}"
SMOKE_UPDATES="${FULL3203_SMOKE_UPDATES:-0}"
EVALUATE_EVERY="${FULL3203_EVALUATE_EVERY:-134}"

SHARDS=(
  "$CACHE_DIR/background_pbg_sigma2_missing_time${TIME_POOL_COUNT}_shard0.h5"
  "$CACHE_DIR/background_pbg_sigma2_missing_time${TIME_POOL_COUNT}_shard1.h5"
  "$CACHE_DIR/background_pbg_sigma2_missing_time${TIME_POOL_COUNT}_shard2.h5"
  "$CACHE_DIR/background_pbg_sigma2_missing_time${TIME_POOL_COUNT}_shard3.h5"
)

cd "$REPO_ROOT"
if [ -z "$PARENT_TERMINAL" ]; then
  PARENT_TERMINAL=$(
    /root/miniconda3/bin/python scripts/select_current_helmholtz_parent.py \
      --results-root "$REPO_ROOT/results" \
      --fallback-terminal "$SEED_TERMINAL" \
      --time-pool-count "$TIME_POOL_COUNT"
  )
fi
echo "Selected audited parent terminal: $PARENT_TERMINAL"
if [ -e "$ARTIFACT_DIR/run_identity.json" ] || [ -e "$ARTIFACT_DIR/terminal.json" ]; then
  echo "Refusing to overwrite an existing full3203 run: $ARTIFACT_DIR" >&2
  exit 64
fi

/root/miniconda3/bin/python - "$PARENT_TERMINAL" "$PRIMARY_CACHE" "$TRAVEL_CACHE" "$TIME_POOL_COUNT" "${SHARDS[@]}" <<'PY'
import sys
from pathlib import Path

from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
from saved_time_phase_operator_v4.eikonal import EikonalTravelCache
from saved_time_phase_operator_v4.multifidelity import fixed_teacher_time_indices
from scripts.diagnose_capacity_ladder_overfit import build_base_config
from scripts.diagnose_helmholtz_g3_heldout import resolve_best_warmstart

terminal, primary, travel, time_pool_count, *shards = sys.argv[1:]
checkpoint, selection = resolve_best_warmstart(terminal)
base = build_base_config(128)
manifest = build_manifest(base.data.source_h5)
if len(manifest.records) != 3203:
    raise SystemExit(f"expected 3203 non-anomaly records, found {len(manifest.records)}")
if "anomaly" not in manifest.excluded_medium_types:
    raise SystemExit("manifest did not audit anomaly as excluded")
provider = BackgroundFieldProvider((primary, *shards))
expected_pool = fixed_teacher_time_indices(
    stored_time_count=len(manifest.time_s), count=int(time_pool_count)
)
if provider.time_indices != expected_pool:
    raise SystemExit(
        f"background common time pool mismatch: {len(provider.time_indices)} != {time_pool_count}"
    )
sample_ids = tuple(record.sample_id for record in manifest.records)
if not provider.covers(sample_ids, expected_pool):
    missing = [sample for sample in sample_ids if not provider.covers((sample,), expected_pool)]
    raise SystemExit(f"background cache set misses {len(missing)} records: {missing[:3]}")
validation_triplet = tuple(
    next(record.sample_id for record in manifest.records
         if record.split == "validation" and record.medium_type == family)
    for family in ("uniform", "layered", "marmousi")
)
if not provider.covers(validation_triplet, range(len(manifest.time_s))):
    raise SystemExit("primary cache cannot support the 401-frame fixed panel gate")
travel_cache = EikonalTravelCache(travel, source_h5=base.data.source_h5)
missing_travel = [sample for sample in sample_ids if sample not in travel_cache.row_by_sample]
if missing_travel:
    raise SystemExit(f"travel cache misses {len(missing_travel)} records: {missing_travel[:3]}")
print({
    "status": "preflight_passed",
    "records": len(sample_ids),
    "time_pool": len(expected_pool),
    "anomaly_excluded": True,
    "parent": str(checkpoint),
    "parent_metric": selection["selection_value"],
})
PY

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
  --warmstart-best-terminal "$PARENT_TERMINAL" \
  --require-best-warmstart \
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
  --training-frames 32 \
  --training-time-pool-from-background \
  --time-adaptive-sampling \
  --time-sampling-uniform-fraction 0.25 \
  --time-residual-ema-momentum 0.3 \
  --validation-frames 32 \
  --per-frame-frame \
  --frame-energy-floor-fraction 0 \
  --uniform-gradient-weight 0.25 \
  --layered-gradient-weight 1.50 \
  --marmousi-gradient-weight 1.25 \
  --uniform-sampling-weight 0.50 \
  --layered-sampling-weight 1.25 \
  --marmousi-sampling-weight 1.25 \
  --local-field-learning-rate 0.0001 \
  --dense-learning-rate 0.00002 \
  --backbone-learning-rate 0.00001 \
  --evaluate-every "$EVALUATE_EVERY" \
  --travel-time-h5 "$TRAVEL_CACHE" \
  --seed 372 \
  --background-cache "$PRIMARY_CACHE" \
  --background-cache-shard "${SHARDS[0]}" \
  --background-cache-shard "${SHARDS[1]}" \
  --background-cache-shard "${SHARDS[2]}" \
  --background-cache-shard "${SHARDS[3]}" \
  --training-splits train validation test_id ood_canonical \
  --records-per-family 0 \
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
