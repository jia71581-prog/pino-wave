#!/usr/bin/env python
"""Train the previously unsupervised grouped dense decoder on train/validation full fields."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from grouped_ufno_mionet import OperatorConfig, GroupedSingleSourceUFNOMIONetOperator
from grouped_ufno_mionet.data.dataset import GroupedWavefieldDataset
from grouped_ufno_mionet.training.checkpoint import load_checkpoint, save_checkpoint

def build_parser():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--dataset',required=True); p.add_argument('--config',default='grouped_ufno_mionet/configs/production.yaml'); p.add_argument('--split',default='train',choices=('train','validation')); p.add_argument('--max-steps',type=int,default=1000); p.add_argument('--device',default='cuda'); return p
def main(argv=None):
 a=build_parser().parse_args(argv); device=a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu'; cfg=OperatorConfig.from_yaml(a.config); model=GroupedSingleSourceUFNOMIONetOperator(cfg).to(device); load_checkpoint(a.checkpoint,model,map_location=device)
 for p in model.parameters(): p.requires_grad_(False)
 for p in model.dense_head.parameters(): p.requires_grad_(True)
 opt=torch.optim.AdamW(model.dense_head.parameters(),lr=2e-4); ds=GroupedWavefieldDataset(a.dataset,split=a.split,frames=cfg.data.frames_per_record,return_full_field=True); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); log=[]
 for step in range(min(a.max_steps,len(ds))):
  item=ds[step]; times=torch.linspace(0.,float(item.pressure_tzx.shape[0]-1)*.0025,8,device=device); idx=torch.linspace(0,item.pressure_tzx.shape[0]-1,8).long(); pred=model.predict_wavefield(item.velocity_mps[None].to(device),item.source_parameters.as_tensor()[None].to(device),times); target=item.pressure_tzx[idx][None].to(device); loss=(pred-target).square().mean(); opt.zero_grad(); loss.backward(); opt.step(); log.append(float(loss.detach()))
 torch.save({'loss':log},out/'metrics.pt'); save_checkpoint(out/'last.pt',model,opt,state={'steps':len(log)},extra={'source_checkpoint':a.checkpoint,'split':a.split})
if __name__=='__main__': main()
