#!/bin/bash
set -uo pipefail
cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
artifact_dir=results/wkb96_pbg_residual_source_r29_long_20260824
log_path="$artifact_dir/launch.log"
mkdir -p "$artifact_dir"
echo "[$(date -u '+%F %T UTC')] r29 P_bg residual long launch supervisor_pid=$$" >> "$log_path"
CUDA_VISIBLE_DEVICES=0 python scripts/train_wkb_frequency_train_panel.py \
  --artifact-dir "$artifact_dir" \
  --preregistration results/wkb96_pbg_residual_source_r29_long_preregistration_20260824.json \
  --base-config configs/grouped_v3/continuous_pilot_w128_legacy_norm_marmousi1_4m_v2.yaml \
  --parent-checkpoint results/wkb96_record_freqsoftmax_r20_continue4000_20260824/checkpoints/update_2600.pt \
  --parent-identity results/wkb96_record_freqsoftmax_r20_continue4000_20260824/run_identity.json \
  --normalization-json /home/jiayh/Data/data/processed/grouped_v3_normalization_marmousi1_4m_v2_checkpoint_compatible.json \
  --travel-time-h5 /home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2.h5 \
  --panel-families marmousi --panel-selection anchor_group_source_holdout \
  --helmholtz-frequencies 96 --helmholtz-frequency-softmax \
  --background-cache-shard artifacts/pbg_sigma2_current_marm_source5_20260824.h5 \
  --zero-initialize-helmholtz-head \
  --updates 2000 --evaluate-every 200 \
  --evaluation-frames 32 --evaluation-time-block 32 --evaluation-macro-records 1 \
  --dense-learning-rate 1e-5 --backbone-learning-rate 5e-6 \
  --local-field-learning-rate 1e-5 --family-gradient-weights marmousi:1 \
  --target-relative-l2 0.05 --seed 372 >> "$log_path" 2>&1
status=$?
echo "[$(date -u '+%F %T UTC')] r29 P_bg residual long exit_code=$status" >> "$log_path"
exit "$status"
