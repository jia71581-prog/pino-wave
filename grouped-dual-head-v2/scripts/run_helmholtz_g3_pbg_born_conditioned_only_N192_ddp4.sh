#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2

exec torchrun --nproc_per_node=4 scripts/diagnose_helmholtz_g3_heldout.py \
  --records-per-family 192 \
  --epochs 40 \
  --background-cache /dev/shm/g3cache/background_pbg_sigma2_g3pool_N192.h5 \
  --normalization-json /root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json \
  --warmstart-helmholtz results/helmholtz_g3_specinj_perbranch_N192_ddp4/checkpoints/update_0768.pt \
  --helmholtz-rank 8 \
  --helmholtz-spectral-bypass \
  --helmholtz-spectral-bypass-per-branch \
  --helmholtz-background-conditioning \
  --helmholtz-background-sigma-cells 2.0 \
  --background-conditioner-only \
  --macro-records 6 \
  --macros-per-update 4 \
  --microbatch-records 1 \
  --backbone-learning-rate 5e-5 \
  --dense-learning-rate 1e-4 \
  --local-field-learning-rate 5e-4 \
  --evaluate-every 96 \
  --seed 372 \
  --artifact-dir results/helmholtz_g3_pbg_born_conditioned_only_v2_N192_ddp4
