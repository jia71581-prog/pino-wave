#!/usr/bin/env bash
set -uo pipefail

ROOT="/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2"
CACHE="$ROOT/results/r38_full_coverage_cache_20260828"
TORCHRUN="/root/miniconda3/bin/torchrun"
PY="/root/miniconda3/bin/python"
cd "$ROOT"

# wait for the R52 orchestrator (single-writer rule)
while pgrep -f run_r52_orchestrator.sh >/dev/null || pgrep -f "torchrun.*train_r52" >/dev/null; do
  sleep 60
done
echo "=== R52 finished, selecting R53 init $(date -Is)"

INIT=$("$PY" - <<'PY'
import json, os
R = "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2/results"
candidates = [
    f"{R}/r39_hfs_tail_finetune_base1e6_hfs1e4_4x500_v3_20260828",
    f"{R}/r52_armA_lr1e5_4x1000_v1_20260829",
    f"{R}/r52_armB_spatial_4x1000_v1_20260829",
]
best, best_score = None, float("inf")
for directory in candidates:
    path = f"{directory}/best.json"
    if not os.path.exists(path):
        continue
    payload = json.load(open(path))
    agg = payload.get("metrics", {}).get("aggregate") or payload.get("aggregate")
    if agg is None:
        continue
    score = float(agg["candidate_max"]) + float(agg["candidate_mean"])
    if score < best_score:
        best, best_score = directory, score
print(f"{best}/best.pt")
PY
)
echo "=== R53 init = $INIT"
test -s "$INIT" || { echo "init checkpoint missing"; exit 2; }

CACHES=(--fit-cache "$CACHE"/fit_shard_{0,1,2,3}.h5 --holdout-cache "$CACHE"/holdout_shard_{0,1,2,3}.h5)
COMMON=(--init-checkpoint "$INIT" --weight-decay 1.0e-4 --hinge-weight 3.0 \
  --gradient-weight 0.05 --num-workers 4 --seed 530829 --amp)
run() { CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 "$@"; }

OUT_S="$ROOT/results/r53_smoke_1x20_v1_20260829"
[[ -e "$OUT_S" ]] && { echo "refusing to reuse $OUT_S"; exit 2; }
echo "=== R53 smoke start $(date -Is)"
run scripts/train_r53_spatiotemporal_tail.py "${CACHES[@]}" "${COMMON[@]}" \
  --output-dir "$OUT_S" --epochs 1 --max-steps-per-epoch 20 --learning-rate 1.0e-5 \
  || { echo "R53 smoke failed"; exit 3; }

"$PY" - "$OUT_S" "$INIT" <<'PY' || { echo "R53 identity gate FAILED"; exit 4; }
import json, sys, torch
initial = json.load(open(sys.argv[1] + "/initial_holdout.json"))["aggregate"]
reference = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
agg = reference["holdout_metrics"]["aggregate"]
dm = abs(initial["candidate_mean"] - agg["candidate_mean"])
dx = abs(initial["candidate_max"] - agg["candidate_max"])
print(f"identity gate: dmean {dm:.2e} dmax {dx:.2e}")
sys.exit(0 if dm <= 2e-4 and dx <= 2e-4 else 1)
PY

OUT="$ROOT/results/r53_spatiotemporal_8ep_v1_20260829"
[[ -e "$OUT" ]] && { echo "refusing to reuse $OUT"; exit 2; }
echo "=== R53 long start $(date -Is)"
run scripts/train_r53_spatiotemporal_tail.py "${CACHES[@]}" "${COMMON[@]}" \
  --output-dir "$OUT" --epochs 8 --learning-rate 1.0e-5
echo "=== R53 long exit $? $(date -Is)"
echo "=== R53 chain done $(date -Is)"
