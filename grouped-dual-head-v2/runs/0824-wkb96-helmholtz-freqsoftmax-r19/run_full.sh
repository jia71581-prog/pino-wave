#!/bin/bash
# r19 FULL — wkb96 Helmholtz + frequency-softmax, continue from r18 best (agg 0.031).
# Single factor vs r18: frequency-softmax gate.  See run_command.txt for the
# ablation contract.  Single GPU (probe is single-device by design); ~28 min.
cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
LOG=results/wkb96_freqsoftmax_r19_continue4000_20260824/launch.log
mkdir -p results/wkb96_freqsoftmax_r19_continue4000_20260824
echo "[$(date '+%F %T')] r19 full launch pid=$$" >> "$LOG"
CUDA_VISIBLE_DEVICES=0 python3 scripts/diagnose_capacity_ladder_overfit.py \
  --artifact-dir results/wkb96_freqsoftmax_r19_continue4000_20260824 \
  --base-config configs/grouped_v3/continuous_pilot_w128_legacy_norm_marmousi1_4m_v2.yaml \
  --width 128 --dense-depth 8 --dense-spectral-rank 112 --dense-modes 32 \
  --local-field --local-field-channels 1,1,2,2 \
  --helmholtz-synthesis --helmholtz-frequencies 96 --helmholtz-rank 0 \
  --helmholtz-frequency-softmax \
  --helmholtz-coefficient-supervision \
  --helmholtz-disable-causal-gate --helmholtz-disable-free-surface-factor \
  --continue-frequency-init-checkpoint results/wkb96_r13_expanded_no_surface_r18_continue3000_20260823/checkpoints/update_3000.pt \
  --continue-frequency-init-identity results/wkb96_r13_expanded_no_surface_r18_continue3000_20260823/run_identity.json \
  --updates 4000 --evaluate-every 100 \
  --dense-learning-rate 1.0e-5 --backbone-learning-rate 5.0e-6 --local-field-learning-rate 1.0e-5 \
  --training-frames 401 --validation-frames 32 --microbatch-records 1 \
  --travel-time-h5 /home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2.h5 \
  --target-aggregate-relative-l2 0.04 --target-family-relative-l2 0.04 \
  --split train --seed 372 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] r19 full EXIT code=$?" >> "$LOG"
