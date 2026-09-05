#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 PREREG CHECKPOINT CACHE MANIFEST BUNDLE TAG" >&2
  exit 2
fi

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1

PREREG=$1
CHECKPOINT=$2
CACHE=$3
MANIFEST=$4
BUNDLE=$5
TAG=$6
SUPERVISOR="results/b2_v10_${TAG}_four_arm_supervisor"
ARMS=(peak cycle025 cycle050 fixed24)

for path in "$PREREG" "$CHECKPOINT" "$CACHE" "$MANIFEST" "$BUNDLE"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required input: $path" >&2
    exit 1
  fi
done
if [[ -e "$SUPERVISOR" ]]; then
  echo "refusing to reuse supervisor directory: $SUPERVISOR" >&2
  exit 1
fi
for arm in "${ARMS[@]}"; do
  output="results/b2_v10_${TAG}_${arm}"
  if [[ -e "$output" || -e "${output}.log" ]]; then
    echo "refusing to reuse arm output: $output" >&2
    exit 1
  fi
done

mkdir -p "$SUPERVISOR"
pids=()
for gpu in 0 1 2 3; do
  arm=${ARMS[$gpu]}
  output="results/b2_v10_${TAG}_${arm}"
  env CUDA_VISIBLE_DEVICES=$gpu python scripts/evaluate_b2_v10_prefix_assimilation.py \
    --arm "$arm" \
    --checkpoint "$CHECKPOINT" \
    --cache "$CACHE" \
    --manifest "$MANIFEST" \
    --bundle "$BUNDLE" \
    --preregistration "$PREREG" \
    --output-dir "$output" > "${output}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$SUPERVISOR/${arm}.pid"
done

exit_codes=()
for pid in "${pids[@]}"; do
  wait "$pid"
  exit_codes+=("$?")
done
python -c 'import json,sys; values=[int(v) for v in sys.argv[1:5]]; json.dump({"status":"complete" if values==[0,0,0,0] else "failed","exit_codes":values},open(sys.argv[5],"w"),indent=2)' \
  "${exit_codes[@]}" "$SUPERVISOR/terminal.json"
for value in "${exit_codes[@]}"; do
  if [[ $value -ne 0 ]]; then
    exit 1
  fi
done
