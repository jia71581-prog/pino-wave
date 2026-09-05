#!/usr/bin/env python3
"""Full-pool phase-WFP capacity comparison on all group-disjoint train roles."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path: sys.path.insert(0, value)

from scripts.train_transfer_dg_wfp_e1 import CacheCollection, FeatureBuilder, atomic_checkpoint, atomic_json, sha256  # noqa: E402
from scripts.train_transfer_dg_wfp_e1d import TravelCollection, apply_phase, evaluate, FAMILIES  # noqa: E402
from saved_time_phase_operator_v4.wfp import BackgroundFrequencyOperator, parameter_count  # noqa: E402


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,action='append',required=True)
    ap.add_argument('--travel',type=Path,action='append',required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--preregistration',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--variant',choices=('base','high'),required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--epochs',type=int,default=2)
    args=ap.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest=json.loads(args.manifest.read_text())
    collection=CacheCollection(args.cache,manifest,expected_count=2800)
    travel=TravelCollection(args.travel,expected_count=2800)
    fit=[i for i,row in enumerate(collection.records) if row[4]=='fit']
    calibration=[i for i,row in enumerate(collection.records) if row[4]=='calibration']
    confirmation=[i for i,row in enumerate(collection.records) if row[4]=='confirmation']
    if (len(fit),len(calibration),len(confirmation))!=(2608,96,96):
        raise RuntimeError('full2800 role census mismatch')
    builder=FeatureBuilder(collection);by_family=defaultdict(list)
    for position in fit: by_family[collection.records[position][3]].append(position)
    width,rank,depth,radii=(32,16,4,(1,2,3,4)) if args.variant=='base' else (64,32,6,(1,2,3,4,5,6))
    device=torch.device('cuda');torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=BackgroundFrequencyOperator(medium_channels=12,source_channels=5,width=width,rank=rank,depth=depth,arm='wfp',radii=radii).to(device)
    per_family_batch=4
    maximum_pairs=max(len(by_family[f])*64 for f in FAMILIES)
    steps_per_epoch=math.ceil(maximum_pairs/per_family_batch)
    total_updates=args.epochs*steps_per_epoch
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_updates,eta_min=2e-6)
    active=torch.from_numpy(builder.active).to(device);rng=np.random.default_rng(args.seed+91)
    identity={'schema':'transfer_dg_wfp_full2800_lane_identity_v1','variant':args.variant,'seed':args.seed,'width':width,'rank':rank,'depth':depth,'epochs':args.epochs,'steps_per_epoch':steps_per_epoch,'total_updates':total_updates,'parameter_count':parameter_count(model),'fit_count':len(fit),'calibration_count':len(calibration),'confirmation_count':len(confirmation),'training_full_truth_allowed':True,'model_input_future_wavefield_frames':0,'test_future_truth_access':False,'manifest_sha256':sha256(args.manifest),'preregistration_sha256':sha256(args.preregistration),'trainer_sha256':sha256(Path(__file__)),'validation_opened':False,'test_id_opened':False}
    atomic_json(identity,args.output_dir/'run_identity.json')
    best=None;global_update=0;started=time.time();metrics_path=args.output_dir/'metrics.jsonl'
    for epoch in range(1,args.epochs+1):
        schedules={}
        for family in FAMILIES:
            values=np.asarray([(p,f) for p in by_family[family] for f in range(64)],dtype=np.int64);rng.shuffle(values);schedules[family]=values
        model.train()
        for step in range(steps_per_epoch):
            selected=[]
            for family in FAMILIES:
                values=schedules[family];indices=(np.arange(per_family_batch)+step*per_family_batch)%len(values);selected.extend(values[indices].tolist())
            items=[builder.build(int(p),int(f),device) for p,f in selected]
            medium,source,scalars,target,aux_target=(torch.cat([item[k] for item in items]) for k in range(5))
            sample_ids=[str(item[5]['sample_id']) for item in items]
            freq=torch.tensor([float(item[5]['frequency_hz']) for item in items],device=device)
            tr=[travel.read(s) for s in sample_ids]
            pt=torch.from_numpy(np.stack([x[0] for x in tr])).to(device);et=torch.from_numpy(np.stack([x[1] for x in tr])).to(device)
            ptotal=torch.tensor([x[2] for x in tr],device=device,dtype=torch.float32);atotal=torch.tensor([x[3] for x in tr],device=device,dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True);prediction,auxiliary=model(medium,source,scalars);prediction,auxiliary=apply_phase(prediction,auxiliary,pt,et,freq,True)
            ploss=(64*(prediction.float()-target.float()).square().sum((1,2,3))/ptotal.clamp_min(1e-8)).mean()
            mask=active[None,None].expand_as(aux_target);aerr=((auxiliary.float()-aux_target.float()).square()*mask).sum((1,2,3));aloss=(64*aerr/atotal.clamp_min(1e-8)).mean();loss=ploss+0.05*aloss
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step();global_update+=1
            if global_update%500==0:
                event={'event':'update','epoch':epoch,'update':global_update,'physical_loss':float(ploss.detach()),'cpml_loss':float(aloss.detach()),'elapsed_s':time.time()-started};metrics_path.open('a').write(json.dumps(event,sort_keys=True)+'\n');print(json.dumps(event),flush=True)
        cal=evaluate(model,builder,travel,calibration,device,True);score=float(cal['physical_mean']);checkpoint={'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'update':global_update,'identity':identity,'calibration':cal};atomic_checkpoint(checkpoint,args.output_dir/'latest.pt')
        if best is None or score<best['score']:
            best={'score':score,'epoch':epoch,'update':global_update,'metrics':cal};atomic_checkpoint(checkpoint,args.output_dir/'best.pt');atomic_json(best,args.output_dir/'best.json')
        metrics_path.open('a').write(json.dumps({'event':'calibration','epoch':epoch,'update':global_update,'metrics':cal},sort_keys=True)+'\n');print(json.dumps({'event':'calibration','epoch':epoch,'score':score}),flush=True)
    checkpoint=torch.load(args.output_dir/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model_state']);model.to(device);confirm=evaluate(model,builder,travel,confirmation,device,True)
    terminal={'schema':'transfer_dg_wfp_full2800_lane_terminal_v1','status':'complete','variant':args.variant,'seed':args.seed,'best_epoch':best['epoch'],'best_update':best['update'],'calibration':best['metrics'],'confirmation':confirm,'best_checkpoint':str((args.output_dir/'best.pt').resolve()),'best_checkpoint_sha256':sha256(args.output_dir/'best.pt'),'elapsed_s':time.time()-started,'training_full_truth_allowed':True,'model_input_future_wavefield_frames':0,'test_future_truth_access':False,'validation_opened':False,'test_id_opened':False};atomic_json(terminal,args.output_dir/'terminal.json');collection.close();travel.close();return 0


if __name__=='__main__': raise SystemExit(main())
