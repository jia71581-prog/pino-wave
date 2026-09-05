#!/usr/bin/env bash
set -euo pipefail

project=/home/jiayh/.config/superpowers/worktrees/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
python=/home/jiayh/miniforge3/envs/PINO/bin/python
v7_artifact=/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v7_eikonal_stage_continuation_pilot_r1
sequence_log=/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v7_arbor/followup_sequence.log

goal_met() {
    local metrics=$1
    [[ -s "$metrics" ]] && jq -s -e '
        any(.[ ];
            (.metrics.aggregate_relative_l2 // 1e9) < 0.10 and
            (.metrics.family_relative_l2.uniform // 1e9) < 0.12 and
            (.metrics.family_relative_l2.layered // 1e9) < 0.12 and
            (.metrics.family_relative_l2.marmousi // 1e9) < 0.12
        )' "$metrics" >/dev/null
}

run_candidate() {
    local config=$1
    local artifact=$2
    local log="$artifact/launcher.log"
    mkdir -p "$artifact"

    if [[ ! -s "$artifact/smoke/terminal.json" ]]; then
        "$python" -u scripts/train_saved_time_v4_full_support.py \
            --config "$config" --smoke-updates 1 >>"$log" 2>&1
    fi
    if [[ -s "$artifact/pilot/terminal.json" ]]; then
        return 0
    fi

    set +e
    "$python" -u scripts/train_saved_time_v4_full_support.py \
        --config "$config" --pilot >>"$log" 2>&1
    local rc=$?
    set -e
    if [[ $rc -eq 2 && -s "$artifact/pilot/terminal.json" ]]; then
        return 0
    fi
    return "$rc"
}

mkdir -p "$(dirname "$sequence_log")"
while [[ ! -s "$v7_artifact/pilot/terminal.json" ]]; do
    state=$(systemctl --user is-active fno-v7-eikonal-stage-continuation-v1.service || true)
    if [[ "$state" == failed ]]; then
        echo "V7 prerequisite service failed" >>"$sequence_log"
        exit 3
    fi
    sleep 30
done

cd "$project"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if goal_met "$v7_artifact/pilot/metrics.jsonl"; then
    echo "target reached by V7; follow-up sequence skipped" >>"$sequence_log"
    exit 0
fi

hybrid_zero_shot=/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v7_arbor/n13_hybrid_zeroshot_epoch3_panel48_frames32.json
if [[ ! -s "$hybrid_zero_shot" ]]; then
    "$python" -u scripts/evaluate_saved_time_v6_scale_sweep.py \
        --config configs/saved_time_v4/v12_hybrid_travel_stage_continuation_pilot.yaml \
        --checkpoint /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v6_muon_warmstart_adamw_r1/run/checkpoints/epoch_0003.pt \
        --checkpoint-identity /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v6_muon_warmstart_adamw_r1/run/run_identity.json \
        --output "$hybrid_zero_shot" \
        --validation-records 48 \
        --validation-frames 32 \
        --panel-epoch 1 \
        --scales 0,1 >>"$sequence_log" 2>&1
fi

configs=(
    configs/saved_time_v4/v12_hybrid_travel_stage_continuation_pilot.yaml
    configs/saved_time_v4/v8_prefix_clip_ray_pilot.yaml
    configs/saved_time_v4/v13_full_backbone_prefix_ray_pilot.yaml
    configs/saved_time_v4/v9_delta_floor_ray_pilot.yaml
    configs/saved_time_v4/v10_coupled_axes_ray_pilot.yaml
    configs/saved_time_v4/v11_family_curriculum_ray_pilot.yaml
)
artifacts=(
    /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v12_hybrid_travel_stage_continuation_pilot_r1
    /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v8_prefix_clip_ray_pilot_r1
    /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v13_full_backbone_prefix_ray_pilot_r1
    /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v9_delta_floor_ray_pilot_r1
    /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v10_coupled_axes_ray_pilot_r1
    /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v11_family_curriculum_ray_pilot_r1
)

for index in "${!configs[@]}"; do
    echo "starting ${configs[$index]}" >>"$sequence_log"
    run_candidate "${configs[$index]}" "${artifacts[$index]}"
    if goal_met "${artifacts[$index]}/pilot/metrics.jsonl"; then
        echo "target reached by ${configs[$index]}" >>"$sequence_log"
        exit 0
    fi
done
echo "controlled follow-up sequence exhausted without reaching target" >>"$sequence_log"
