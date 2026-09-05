#!/usr/bin/env bash
set -euo pipefail

project=/home/jiayh/.config/superpowers/worktrees/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
python=/home/jiayh/miniforge3/envs/PINO/bin/python
arbor=/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v7_arbor
artifact=/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v7_eikonal_stage_continuation_pilot_r1
scale_output="$arbor/n5_scale_sweep_epoch3_extended_panel48_frames32.json"
eikonal_output="$arbor/n8_eikonal_zeroshot_epoch3_panel48_frames32.json"
log="$artifact/launcher.log"
config=configs/saved_time_v4/v7_eikonal_stage_continuation_pilot.yaml
checkpoint=/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v6_muon_warmstart_adamw_r1/run/checkpoints/epoch_0003.pt
checkpoint_identity=/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v6_muon_warmstart_adamw_r1/run/run_identity.json

mkdir -p "$artifact"
while [[ ! -s "$scale_output" ]]; do
    state=$(systemctl --user is-active fno-v6-epoch3-scale-gate-robust.service || true)
    if [[ "$state" == failed ]]; then
        echo "scale gate failed before producing $scale_output" >>"$log"
        exit 3
    fi
    sleep 20
done

cd "$project"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -s "$eikonal_output" ]]; then
    "$python" -u scripts/evaluate_saved_time_v6_scale_sweep.py \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --checkpoint-identity "$checkpoint_identity" \
        --output "$eikonal_output" \
        --validation-records 48 \
        --validation-frames 32 \
        --panel-epoch 1 \
        --scales 0,1 >>"$log" 2>&1
fi

if [[ ! -s "$artifact/smoke/terminal.json" ]]; then
    "$python" -u scripts/train_saved_time_v4_full_support.py \
        --config "$config" --smoke-updates 1 >>"$log" 2>&1
fi

if [[ -s "$artifact/pilot/terminal.json" ]]; then
    status=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "$artifact/pilot/terminal.json")
    if [[ "$status" == complete || "$status" == pilot_gate_failed ]]; then
        exit 0
    fi
fi

set +e
"$python" -u scripts/train_saved_time_v4_full_support.py \
    --config "$config" --pilot >>"$log" 2>&1
rc=$?
set -e

# Exit code 2 is the documented scientific gate result, not an execution failure.
if [[ $rc -eq 2 && -s "$artifact/pilot/terminal.json" ]]; then
    status=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "$artifact/pilot/terminal.json")
    if [[ "$status" == pilot_gate_failed ]]; then
        exit 0
    fi
fi
exit "$rc"
