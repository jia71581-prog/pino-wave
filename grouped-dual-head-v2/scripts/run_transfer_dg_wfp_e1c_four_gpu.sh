#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
out="$root/results/transfer_dg_wfp_e1c_phase_carrier_20260902"
manifest="$root/results/transfer_dg_wfp_e1_manifest_256_20260902.json"
prereg="$root/results/transfer_dg_wfp_e1c_phase_carrier_preregistration_20260902.json"
cache="$root/results/transfer_dg_wfp_e1_cache_20260902"
mkdir "$out"

carriers=(raw raw phase phase)
seeds=(372 733 372 733)
pids=()
for gpu in 0 1 2 3; do
  carrier=${carriers[$gpu]}
  seed=${seeds[$gpu]}
  lane="$out/${carrier}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH="$root:$root/src" \
    python "$root/scripts/diagnose_transfer_dg_wfp_phase_carrier.py" \
      --cache "$cache/shard_0.h5" --cache "$cache/shard_1.h5" \
      --cache "$cache/shard_2.h5" --cache "$cache/shard_3.h5" \
      --manifest "$manifest" --preregistration "$prereg" \
      --output-dir "$lane" --carrier "$carrier" --seed "$seed" \
      > "$out/${carrier}_s${seed}.log" 2>&1 &
  pids+=("$!")
done

codes=()
status=complete
for pid in "${pids[@]}"; do
  if wait "$pid"; then codes+=(0); else codes+=("$?"); status=rejected; fi
done
python - "$out" "$status" "${codes[@]}" <<'PY'
import json, os, sys
from pathlib import Path
out=Path(sys.argv[1]); status=sys.argv[2]; codes=[int(x) for x in sys.argv[3:]]
rows=[]
for carrier in ('raw','phase'):
  for seed in (372,733): rows.append(json.loads((out/f'{carrier}_s{seed}/terminal.json').read_text()))
means={carrier:sum(float(r['best']['metrics']['relative_l2']) for r in rows if r['carrier']==carrier)/2 for carrier in ('raw','phase')}
phase=[r for r in rows if r['carrier']=='phase']
accepted=(all(r['status']=='passed' for r in phase) and means['phase']<means['raw'])
summary={'schema':'transfer_dg_wfp_e1c_phase_carrier_summary_v1','status':'accepted' if accepted else 'rejected','decision':'rerun_256_with_phase_carrier' if accepted else 'close_direct_phase_carrier','means':means,'relative_gain':1-means['phase']/means['raw'],'lanes':{f"{r['carrier']}_s{r['seed']}":r for r in rows},'validation_opened':False,'test_id_opened':False}
tmp=out/f'summary.json.partial.{os.getpid()}';tmp.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');os.replace(tmp,out/'summary.json')
terminal={'schema':'transfer_dg_wfp_e1c_supervisor_terminal_v1','status':'complete','return_codes':codes,'summary':str(out/'summary.json'),'validation_opened':False,'test_id_opened':False}
tmp=out/f'terminal.json.partial.{os.getpid()}';tmp.write_text(json.dumps(terminal,indent=2,sort_keys=True)+'\n');os.replace(tmp,out/'terminal.json')
print(json.dumps(terminal,indent=2,sort_keys=True))
PY
