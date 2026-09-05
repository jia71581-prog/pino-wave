#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
export PYTHONPATH=src:.
exec /root/miniconda3/bin/python -u scripts/probe_train_only_travel_time_warp_capacity.py \
  --dataset /root/autodl-tmp/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_train_marmousi_hires_v5_dt1e5_t401_mixed/dataset_v5_dt1e5_t401_mixed.h5 \
  --travel-cache /root/autodl-tmp/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_train_marmousi_hires_v5_dt1e5_t401_mixed/hybrid_travel_train_validation_v5.h5 \
  --output results/train_only_travel_time_warp_capacity_r1.json \
  --source-indices 2280,2326,2574,2237,2407,2286 \
  --expected-sample-sha256 409bd1fc0646dfcdf39be4e325b1f0d16082760a0ad735b5f93a9cd331f2464b,b6ef006243c7cdeb2f378ccb8fba8ee4c35b66d50d50dbbc3ea49edc859ccd4e,3d9a7603d6fb24c2a18f51831693cd2abc94dce4ed41b74b9930ee3a459789b1,3c01a480d0947178c304015f0dc44c5c232ce4c0b9ed4159a7ba4b0f85d8c8f7,066dce13e9802016b253bd98e5a0b345a1324de6ce61ecd57e3c4e94235437ff,bd430a1b4ba76a0be636c6cacdd6cf193d0154571daef17daf32ce71771dde59 \
  --warp-scales=-0.04,-0.03,-0.02,-0.015,-0.01,-0.0075,-0.005,-0.0025,0.0,0.0025,0.005,0.0075,0.01,0.015,0.02,0.03,0.04 \
  --route-threshold-hz 24 \
  --candidate-fraction 0.45 \
  --internal-dt-s 0.00025 \
  --npml 20 \
  --threads 32
