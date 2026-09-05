#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
CONFIG_PATH="$PROJECT_ROOT/configs/datasets/acoustic_lwc84_2km_401x401_to_201_marmousi1_v5_dt1e5_t401.yaml"
NEW_ROOT=/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_v5_dt1e5_t401
NEW_DATASET="$NEW_ROOT/dataset_marmousi1_v5_dt1e5_t401.h5"
OLD_ROOT=/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1
OLD_DATASET="$OLD_ROOT/dataset_v1.invalidated_source_for_hybrid_v2.h5"
OLD_MANIFEST="$OLD_ROOT/manifest.jsonl"
HYBRID_ROOT=/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_train_marmousi_hires_v5_dt1e5_t401_mixed
HYBRID_DATASET="$HYBRID_ROOT/dataset_v5_dt1e5_t401_mixed.h5"
HYBRID_MANIFEST="$HYBRID_ROOT/manifest_v5_dt1e5_t401_mixed.jsonl"
PIPELINE_ROOT="$PROJECT_ROOT/results/marmousi1_v5_dt1e5_t401_pipeline_20260808"

mkdir -p "$PIPELINE_ROOT" "$HYBRID_ROOT"
cd "$PROJECT_ROOT"

# A stale terminal marker from an interrupted/restarted launcher must never be
# mistaken for completion while this resumable production run is active.
rm -f "$PIPELINE_ROOT/terminal.status" "$PIPELINE_ROOT/terminal.exitcode"

on_exit() {
  pipeline_code=$?
  printf '%s\n' "$pipeline_code" > "$PIPELINE_ROOT/terminal.exitcode"
  if [[ "$pipeline_code" -eq 0 ]]; then
    printf '%s\n' COMPLETE > "$PIPELINE_ROOT/terminal.status"
  else
    printf '%s\n' FAILED > "$PIPELINE_ROOT/terminal.status"
  fi
}
trap on_exit EXIT

# This run is intentionally allowed to coexist with the active Helmholtz job.
# Batch 64 is expected to use about 6.6 GiB per GPU from the observed batch-40
# footprint, filling the devices while retaining roughly 1.4 GiB headroom.
printf '%s starting_concurrent_700_sample_generation batch_size=64\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

printf '%s starting_700_sample_generation\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python scripts/launch_gpu_dataset_workers.py \
  --config "$CONFIG_PATH" \
  --splits train \
  --medium-types marmousi \
  --gpu-ids 0,1,2,3 \
  --output "$NEW_ROOT" \
  --confirm-production RUN_MARMOUSI1_V5_DT1E5_T401_700 \
  --batch-size 64 \
  --no-cpu-fallback \
  --resume \
  --frozen-manifest

printf '%s validating_700_sample_dataset\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ ! -f "$NEW_ROOT/strict_same_protocol_validation.json" ]]; then
  python scripts/validate_acoustic_dataset.py \
    --dataset "$NEW_DATASET" \
    --config "$CONFIG_PATH" \
    --expected-samples 700 \
    --strict \
    --readback-check \
    --full \
    --output "$NEW_ROOT/strict_same_protocol_validation.json"
fi

printf '%s building_mixed_internal_dt_hybrid_vds\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ ! -f "$HYBRID_ROOT/replacement_report.json" ]]; then
  python scripts/build_lwc84_train_marmousi_replacement_vds.py \
    --old-dataset "$OLD_DATASET" \
    --old-manifest "$OLD_MANIFEST" \
    --new-marmousi-dataset "$NEW_DATASET" \
    --new-manifest "$NEW_ROOT/manifest.jsonl" \
    --output-dataset "$HYBRID_DATASET" \
    --output-manifest "$HYBRID_MANIFEST" \
    --report "$HYBRID_ROOT/replacement_report.json" \
    --expected-replacements 700 \
    --allow-mixed-internal-dt
fi

printf '%s computing_train_normalization_stats\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ ! -f "$HYBRID_ROOT/normalization_stats_v5.json" ]]; then
  python scripts/compute_lwc84_train_stats.py \
    --dataset "$HYBRID_DATASET" \
    --output "$HYBRID_ROOT/normalization_stats_v5.json" \
    --expected-train-samples 2800 \
    --time-chunk-size 16
fi

printf '%s running_final_mixed_protocol_gate\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ ! -f "$HYBRID_ROOT/mixed_protocol_replacement_gate.json" ]]; then
  python scripts/gate_lwc84_marmousi_replacement.py \
    --old-dataset "$OLD_DATASET" \
    --new-marmousi-dataset "$NEW_DATASET" \
    --hybrid-dataset "$HYBRID_DATASET" \
    --hybrid-manifest "$HYBRID_MANIFEST" \
    --replacement-report "$HYBRID_ROOT/replacement_report.json" \
    --new-strict-report "$NEW_ROOT/strict_same_protocol_validation.json" \
    --normalization-stats "$HYBRID_ROOT/normalization_stats_v5.json" \
    --output "$HYBRID_ROOT/mixed_protocol_replacement_gate.json" \
    --expected-samples 4003 \
    --expected-replacements 700 \
    --expected-train-samples 2800 \
    --allow-mixed-internal-dt
fi

sha256sum \
  "$NEW_DATASET" \
  "$HYBRID_DATASET" \
  "$HYBRID_MANIFEST" \
  "$HYBRID_ROOT/normalization_stats_v5.json" \
  "$HYBRID_ROOT/mixed_protocol_replacement_gate.json" \
  > "$HYBRID_ROOT/sha256sums.txt"

printf '%s pipeline_complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
