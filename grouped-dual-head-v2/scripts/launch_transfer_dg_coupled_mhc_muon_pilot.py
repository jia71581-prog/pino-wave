#!/usr/bin/env python3
"""Launch the four-arm persistent-pyramid/MoE train-only factorial pilot."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/'results'
PREREG=RESULTS/'transfer_dg_coupled_pyramid_moe_pilot_preregistration_20260903.json'
PILOT=RESULTS/'transfer_dg_coupled_mhc_muon_bt_ufno_pilot_manifest_20260903.json'
FULL=RESULTS/'transfer_dg_wfp_full2800_manifest_20260902.json'
PARENT=RESULTS/'transfer_dg_wfp_full2800_final_ddp4_20260903/latest.pt'
RESIDUAL=RESULTS/'transfer_dg_phase_scatter64_full_20260903/cache'
BASE=RESULTS/'transfer_dg_wfp_full2800_cache_20260902'
TRAVEL=RESULTS/'transfer_dg_wfp_full2800_travel_20260902'
TRAINER=ROOT/'scripts/train_transfer_dg_coupled_mhc_muon_pilot.py'
OUT=RESULTS/'transfer_dg_coupled_pyramid_moe_pilot_20260903'
ARMS=(('plain_adamw',False,'adamw'),('mhc_adamw',True,'adamw'),('plain_muon',False,'muon'),('mhc_muon',True,'muon'))

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(payload,path):
 tmp=path.with_name(f'{path.name}.partial.{os.getpid()}');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)

def main():
 if OUT.exists():raise FileExistsError(OUT)
 prereg=json.loads(PREREG.read_text());b=prereg['bindings']
 for key,path in {'trainer_sha256':TRAINER,'model_sha256':ROOT/'saved_time_phase_operator_v4/coupled_pyramid_moe_wave.py','base_model_sha256':ROOT/'saved_time_phase_operator_v4/coupled_mhc_wave.py','launcher_sha256':Path(__file__),'pilot_manifest_sha256':PILOT,'parent_checkpoint_sha256':PARENT}.items():
  observed=sha256(path)
  if observed!=b[key]:raise RuntimeError(f'binding drift {key}: {observed}')
 OUT.mkdir(parents=True);processes=[];logs=[];commands=[]
 residual_args=sum((['--residual-cache',str(RESIDUAL/f'shard_{i}.h5')] for i in range(4)),[])
 base_args=sum((['--base-cache',str(BASE/f'shard_{i}.h5')] for i in range(4)),[])
 travel_args=sum((['--travel',str(TRAVEL/f'shard_{i}.h5')] for i in range(4)),[])
 started=time.time()
 for gpu,(name,mhc,optimizer) in enumerate(ARMS):
  command=[sys.executable,str(TRAINER),*residual_args,*base_args,*travel_args,'--full-manifest',str(FULL),'--pilot-manifest',str(PILOT),'--preregistration',str(PREREG),'--parent-checkpoint',str(PARENT),'--output-dir',str(OUT/name),'--optimizer',optimizer,'--seed',str(prereg['training']['seed'])]
  if mhc:command.append('--use-mhc')
  env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'HDF5_USE_FILE_LOCKING':'FALSE','OMP_NUM_THREADS':'8','PYTHONPATH':f'{ROOT}:{ROOT/"src"}'})
  log=(OUT/f'{name}.log').open('w');logs.append(log);processes.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT));commands.append(command)
 atomic({'schema':'transfer_dg_coupled_mhc_muon_pilot_identity_v1','pid':os.getpid(),'child_pids':[p.pid for p in processes],'arms':[a[0] for a in ARMS],'commands':commands,'gpu_ids':[0,1,2,3],'validation_opened':False,'test_id_opened':False,'started_unix_s':started},OUT/'run_identity.json')
 codes=[p.wait() for p in processes]
 for log in logs:log.close()
 rows={}
 for name,_,_ in ARMS:
  path=OUT/name/'terminal.json'
  if path.exists():rows[name]=json.loads(path.read_text())
 if len(rows)==4 and all(c==0 for c in codes):
  scores={name:r['confirmation']['candidate_mean'] for name,r in rows.items()};parent=next(iter(rows.values()))['confirmation']['parent_mean'];winner=min(scores,key=scores.get);accepted=scores[winner]<parent
  summary={'schema':'transfer_dg_coupled_mhc_muon_pilot_summary_v1','status':'accepted' if accepted else 'rejected','winner':winner,'parent_confirmation_mean':parent,'candidate_confirmation_means':scores,'winner_relative_gain':1-scores[winner]/parent,'decision':'replicate_winner' if accepted else 'reject_full_network_route','arms':rows,'validation_opened':False,'test_id_opened':False}
 else:summary={'schema':'transfer_dg_coupled_mhc_muon_pilot_summary_v1','status':'failed','return_codes':codes,'terminal_count':len(rows),'validation_opened':False,'test_id_opened':False}
 atomic(summary,OUT/'summary.json');atomic({'schema':'transfer_dg_coupled_mhc_muon_pilot_terminal_v1','status':'complete' if len(rows)==4 and all(c==0 for c in codes) else 'failed','return_codes':codes,'summary':str((OUT/'summary.json').resolve()),'elapsed_s':time.time()-started,'validation_opened':False,'test_id_opened':False},OUT/'terminal.json')
 return 0 if len(rows)==4 and all(c==0 for c in codes) else 2

if __name__=='__main__':raise SystemExit(main())
