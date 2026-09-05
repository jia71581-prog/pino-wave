#!/usr/bin/env bash
set -uo pipefail

ROOT="/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2"
CACHE="$ROOT/results/r38_full_coverage_cache_20260828"
INIT="$ROOT/results/r39_hfs_tail_finetune_base1e6_hfs1e4_4x500_v3_20260828/best.pt"
TORCHRUN="/root/miniconda3/bin/torchrun"
cd "$ROOT"

CACHES=(--fit-cache "$CACHE"/fit_shard_{0,1,2,3}.h5 --holdout-cache "$CACHE"/holdout_shard_{0,1,2,3}.h5)
COMMON=(--init-checkpoint "$INIT" --batch-size 16 --eval-batch-size 16 \
  --weight-decay 1.0e-4 --hinge-weight 3.0 --gradient-weight 0.05 --late-start-s 0.50 --amp)

run() { CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 "$@"; }

OUT_A="$ROOT/results/r52_armA_lr1e5_4x1000_v1_20260829"
[[ -e "$OUT_A" ]] && { echo "refusing to reuse $OUT_A"; exit 2; }
echo "=== ARM A start $(date -Is)"
run scripts/train_r52_hfs_lr10.py "${CACHES[@]}" "${COMMON[@]}" \
  --output-dir "$OUT_A" --epochs 4 --max-steps-per-epoch 1000 \
  --learning-rate 1.0e-5 --seed 520829
STATUS_A=$?
echo "=== ARM A exit $STATUS_A $(date -Is)"

# frozen rule: B uses 1e-5 unless arm A overshoots (any epoch mean >= 0.0250) or failed
B_LR="1.0e-5"
if [[ $STATUS_A -ne 0 ]] || ! /root/miniconda3/bin/python - "$OUT_A" <<'PY'
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1] + "/holdout_metrics.jsonl")]
except OSError:
    sys.exit(1)
sys.exit(0 if rows and all(r["aggregate"]["candidate_mean"] < 0.0250 for r in rows) else 1)
PY
then B_LR="1.0e-6"; fi
echo "=== ARM B base lr = $B_LR"

OUT_BS="$ROOT/results/r52_armB_smoke_1x20_v1_20260829"
[[ -e "$OUT_BS" ]] && { echo "refusing to reuse $OUT_BS"; exit 2; }
echo "=== ARM B smoke start $(date -Is)"
run scripts/train_r52_spatial_hfs.py "${CACHES[@]}" "${COMMON[@]}" \
  --output-dir "$OUT_BS" --epochs 1 --max-steps-per-epoch 20 \
  --learning-rate "$B_LR" --seed 520830 || { echo "ARM B smoke failed"; exit 3; }

/root/miniconda3/bin/python - "$OUT_BS" <<'PY' || { echo "ARM B identity gate FAILED"; exit 4; }
import json, sys
a = json.load(open(sys.argv[1] + "/initial_holdout.json"))["aggregate"]
diff = abs(a["candidate_max"] - 0.06582062992168616)
print(f"identity gate: initial max {a['candidate_max']:.8f} diff {diff:.2e}")
sys.exit(0 if diff <= 1e-4 else 1)
PY

OUT_B="$ROOT/results/r52_armB_spatial_4x1000_v1_20260829"
[[ -e "$OUT_B" ]] && { echo "refusing to reuse $OUT_B"; exit 2; }
echo "=== ARM B start $(date -Is)"
run scripts/train_r52_spatial_hfs.py "${CACHES[@]}" "${COMMON[@]}" \
  --output-dir "$OUT_B" --epochs 4 --max-steps-per-epoch 1000 \
  --learning-rate "$B_LR" --seed 520830
echo "=== ARM B exit $? $(date -Is)"
echo "=== R52 orchestrator done $(date -Is)"
