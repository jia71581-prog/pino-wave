#!/usr/bin/env python3
"""Auto-launch the full-pool paired pretraining after cache/travel completion."""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'results/transfer_dg_wfp_full2800_cache_20260902'
TRAVEL=ROOT/'results/transfer_dg_wfp_full2800_travel_20260902'
TRAVEL_TERMINAL=ROOT/'results/transfer_dg_wfp_full2800_travel_terminal_20260902.json'
OUT=ROOT/'results/transfer_dg_wfp_full2800_pretraining_20260902'
MANIFEST=ROOT/'results/transfer_dg_wfp_full2800_manifest_20260902.json'
PREREG=ROOT/'results/transfer_dg_wfp_full2800_pretraining_preregistration_20260902.json'
LANES=(('base',372),('base',733),('high',372),('high',733))

def atomic(payload,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'{path.name}.partial.{os.getpid()}');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)

def main():
    queue=ROOT/'results/transfer_dg_wfp_full2800_pretraining_queue_20260902.json';atomic({'schema':'transfer_dg_wfp_full2800_pretraining_queue_v1','pid':os.getpid(),'state':'waiting_for_travel','self_approved':True,'training_full_truth_allowed':True,'test_future_truth_access':False,'validation_opened':False,'test_id_opened':False},queue)
    while not TRAVEL_TERMINAL.is_file(): time.sleep(60)
    prerequisite=json.loads(TRAVEL_TERMINAL.read_text())
    if prerequisite.get('status')!='complete' or int(prerequisite.get('record_count',-1))!=2800:
        atomic({'schema':'transfer_dg_wfp_full2800_pretraining_terminal_v1','status':'blocked_by_travel','prerequisite':prerequisite},ROOT/'results/transfer_dg_wfp_full2800_pretraining_terminal_20260902.json');return 2
    OUT.mkdir(parents=True,exist_ok=False);processes=[];logs=[]
    cache_args=sum((['--cache',str(CACHE/f'shard_{i}.h5')] for i in range(4)),[]);travel_args=sum((['--travel',str(TRAVEL/f'shard_{i}.h5')] for i in range(4)),[])
    for gpu,(variant,seed) in enumerate(LANES):
        command=[sys.executable,str(ROOT/'scripts/train_transfer_dg_wfp_full2800.py'),*cache_args,*travel_args,'--manifest',str(MANIFEST),'--preregistration',str(PREREG),'--output-dir',str(OUT/f'{variant}_s{seed}'),'--variant',variant,'--seed',str(seed)]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['HDF5_USE_FILE_LOCKING']='FALSE';env['PYTHONPATH']=f'{ROOT}:{ROOT/"src"}'
        log=(OUT/f'{variant}_s{seed}.log').open('w');processes.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT));logs.append(log)
    atomic({'schema':'transfer_dg_wfp_full2800_pretraining_identity_v1','pid':os.getpid(),'child_pids':[p.pid for p in processes],'self_approved':True,'training_full_truth_allowed':True,'test_future_truth_access':False,'validation_opened':False,'test_id_opened':False},OUT/'run_identity.json')
    codes=[p.wait() for p in processes]
    for log in logs:log.close()
    rows=[json.loads((OUT/f'{v}_s{s}/terminal.json').read_text()) for v,s in LANES if (OUT/f'{v}_s{s}/terminal.json').is_file()]
    if len(rows)==4:
        means={v:sum(float(r['confirmation']['physical_mean']) for r in rows if r['variant']==v)/2 for v in ('base','high')};accepted=all(c==0 for c in codes) and means['high']<means['base'];summary={'schema':'transfer_dg_wfp_full2800_pretraining_summary_v1','status':'accepted' if accepted else 'rejected','decision':'auto_approve_final_all2800_retrain' if accepted else 'stop_before_validation','means':means,'relative_gain':1-means['high']/means['base'],'lanes':{f"{r['variant']}_s{r['seed']}":r for r in rows},'self_approved':True,'validation_opened':False,'test_id_opened':False}
    else:summary={'schema':'transfer_dg_wfp_full2800_pretraining_summary_v1','status':'failed','return_codes':codes,'validation_opened':False,'test_id_opened':False}
    atomic(summary,OUT/'summary.json');atomic({'schema':'transfer_dg_wfp_full2800_pretraining_terminal_v1','status':'complete' if len(rows)==4 else 'failed','return_codes':codes,'summary':str(OUT/'summary.json'),'validation_opened':False,'test_id_opened':False},OUT/'terminal.json');return 0 if len(rows)==4 else 2

if __name__=='__main__':raise SystemExit(main())
