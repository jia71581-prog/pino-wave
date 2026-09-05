#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
out="$root/results/transfer_dg_wfp_e1d_v2_generalization_20260902"
manifest="$root/results/transfer_dg_wfp_e1_manifest_256_20260902.json"
prereg="$root/results/transfer_dg_wfp_e1d_v2_generalization_preregistration_20260902.json"
cache="$root/results/transfer_dg_wfp_e1_cache_20260902"
travel="$root/results/transfer_dg_wfp_e1d_travel_v2_20260902"
mkdir "$out"

carriers=(raw raw phase phase)
seeds=(372 733 372 733)
pids=()
for gpu in 0 1 2 3; do
  carrier=${carriers[$gpu]}
  seed=${seeds[$gpu]}
  lane="$out/${carrier}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH="$root:$root/src" \
    python "$root/scripts/train_transfer_dg_wfp_e1d.py" \
      --cache "$cache/shard_0.h5" --cache "$cache/shard_1.h5" \
      --cache "$cache/shard_2.h5" --cache "$cache/shard_3.h5" \
      --travel "$travel/shard_0.h5" --travel "$travel/shard_1.h5" \
      --travel "$travel/shard_2.h5" --travel "$travel/shard_3.h5" \
      --manifest "$manifest" --preregistration "$prereg" \
      --output-dir "$lane" --carrier "$carrier" --seed "$seed" \
      > "$out/${carrier}_s${seed}.log" 2>&1 &
  pids+=("$!")
done
python - "$out" "${pids[@]}" <<'PY'
import json,os,sys
from pathlib import Path
out=Path(sys.argv[1]);payload={'schema':'transfer_dg_wfp_e1d_supervisor_identity_v1','supervisor_pid':os.getppid(),'child_pids':[int(x) for x in sys.argv[2:]],'training_full_truth_allowed':True,'model_input_future_wavefield_frames':0,'test_future_truth_access':False,'validation_opened':False,'test_id_opened':False}
p=out/'run_identity.json';p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

codes=()
for pid in "${pids[@]}"; do
  if wait "$pid"; then codes+=(0); else codes+=("$?"); fi
done
python - "$out" "${codes[@]}" <<'PY'
import json,os,sys
from pathlib import Path
out=Path(sys.argv[1]);codes=[int(x) for x in sys.argv[2:]]
rows=[]
for carrier in ('raw','phase'):
  for seed in (372,733): rows.append(json.loads((out/f'{carrier}_s{seed}/terminal.json').read_text()))
means={carrier:sum(float(r['confirmation']['physical_mean']) for r in rows if r['carrier']==carrier)/2 for carrier in ('raw','phase')}
family={carrier:{fam:sum(float(r['confirmation']['per_family'][fam]) for r in rows if r['carrier']==carrier)/2 for fam in ('uniform','layered','anomaly','marmousi')} for carrier in ('raw','phase')}
boundary=all(float(r['confirmation']['top_pressure_max_abs'])==0 and float(r['confirmation']['outer_pressure_max_abs'])==0 for r in rows)
accepted=(all(c==0 for c in codes) and means['phase']<means['raw'] and boundary)
summary={'schema':'transfer_dg_wfp_e1d_generalization_summary_v1','status':'accepted' if accepted else 'rejected','decision':'expand_to_all_2800' if accepted else 'stop_before_all_2800','means':means,'per_family_two_seed_mean':family,'relative_gain':1-means['phase']/means['raw'],'boundary_gate_passed':boundary,'return_codes':codes,'lanes':{f"{r['carrier']}_s{r['seed']}":{'best_epoch':r['best_epoch'],'best_update':r['best_update'],'calibration':r['calibration'],'confirmation':r['confirmation'],'checkpoint':r['best_checkpoint'],'checkpoint_sha256':r['best_checkpoint_sha256']} for r in rows},'training_full_truth_allowed':True,'model_input_future_wavefield_frames':0,'test_wavefield_access':'registered_early_prefix_only','test_future_truth_access':False,'validation_opened':False,'test_id_opened':False}
tmp=out/f'summary.json.partial.{os.getpid()}';tmp.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');os.replace(tmp,out/'summary.json')
terminal={'schema':'transfer_dg_wfp_e1d_supervisor_terminal_v1','status':'complete' if all(c==0 for c in codes) else 'failed','return_codes':codes,'summary':str(out/'summary.json'),'validation_opened':False,'test_id_opened':False}
tmp=out/f'terminal.json.partial.{os.getpid()}';tmp.write_text(json.dumps(terminal,indent=2,sort_keys=True)+'\n');os.replace(tmp,out/'terminal.json')
print(json.dumps(terminal,indent=2,sort_keys=True))
PY
