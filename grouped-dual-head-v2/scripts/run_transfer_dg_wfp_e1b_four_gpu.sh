#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
out="$root/results/transfer_dg_wfp_e1b_loss_capacity_20260902"
manifest="$root/results/transfer_dg_wfp_e1_manifest_256_20260902.json"
prereg="$root/results/transfer_dg_wfp_e1b_loss_capacity_preregistration_20260902.json"
cache="$root/results/transfer_dg_wfp_e1_cache_20260902"
mkdir "$out"

arms=(fno fno wfp wfp)
seeds=(372 733 372 733)
pids=()
for gpu in 0 1 2 3; do
  arm=${arms[$gpu]}
  seed=${seeds[$gpu]}
  lane="$out/${arm}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH="$root:$root/src" \
    python "$root/scripts/diagnose_transfer_dg_wfp_loss_capacity.py" \
      --cache "$cache/shard_0.h5" --cache "$cache/shard_1.h5" \
      --cache "$cache/shard_2.h5" --cache "$cache/shard_3.h5" \
      --manifest "$manifest" --preregistration "$prereg" \
      --output-dir "$lane" --arm "$arm" --seed "$seed" \
      > "$out/${arm}_s${seed}.log" 2>&1 &
  pids+=("$!")
done

codes=()
status=complete
for pid in "${pids[@]}"; do
  if wait "$pid"; then codes+=(0); else codes+=("$?"); status=failed; fi
done
python - "$out" "$status" "${codes[@]}" <<'PY'
import json, os, sys
from pathlib import Path
out=Path(sys.argv[1]); status=sys.argv[2]; codes=[int(x) for x in sys.argv[3:]]
payload={'schema':'transfer_dg_wfp_e1b_supervisor_terminal_v1','status':status,'return_codes':codes,'validation_opened':False,'test_id_opened':False}
tmp=out/f'terminal.json.partial.{os.getpid()}';tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(tmp,out/'terminal.json')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
