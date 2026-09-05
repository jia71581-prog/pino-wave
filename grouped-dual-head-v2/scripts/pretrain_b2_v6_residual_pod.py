#!/usr/bin/env python3
"""Offline train-truth pretraining of family POD modes and encoding priors."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
for v in (str(ROOT),str(ROOT/'src')):
    if v not in sys.path: sys.path.insert(0,v)
from saved_time_phase_operator_v4.instance_adaptation.b2_v6_pod import fit_residual_pod,pod_coefficients,fit_ridge_prior,summary_encoding_features
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator
FAMILIES=('uniform','layered','marmousi')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def atom(x,p):
 q=p.with_name(f'{p.name}.partial.{os.getpid()}');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def main():
 p=argparse.ArgumentParser();
 for n in ('checkpoint','cache','manifest','preregistration','output_dir'): p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
 p.add_argument('--rank',type=int,default=4);a=p.parse_args();out=a.output_dir
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);terminal=out/'terminal.json'
 try:
  pre=json.load(open(a.preregistration));b=pre['bindings']
  for path,key in ((Path(__file__),'pretrain_sha256'),(a.checkpoint,'checkpoint_sha256'),(a.cache,'fit_cache_sha256'),(a.manifest,'fit_manifest_sha256')):
   if sha(path)!=b[key]:raise RuntimeError(f'binding drift {path}')
  m=json.load(open(a.manifest));ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);ident=ck['identity'];dev=torch.device('cuda')
  with h5py.File(a.cache,'r',swmr=True) as h:
   base=torch.from_numpy(h['base_seq'][:].astype(np.float32))[:,:,None];target=torch.from_numpy(h['target'][:].astype(np.float32))[:,:,None];cond=torch.from_numpy(h['cond'][:].astype(np.float32));families=h['family'][:].astype(str)
  model=FrameConditionedPropagator(state_channels=8,cond_channels=7,width=ident['width'],spectral_rank=ident['spectral_rank'],modes=24,depth=4,gate_init=1.,activation_checkpointing=False).to(dev);model.load_state_dict(ck['model_state']);model.eval()
  residuals=defaultdict(list);targets=defaultdict(list);features={e:defaultdict(list) for e in ('E0','E1','E2')}
  with h5py.File(m['source_h5'],'r',swmr=True) as src:
   for i,row in enumerate(m['records']):
    bs=base[i:i+1].to(dev);cd=cond[i:i+1].to(dev);tg=target[i:i+1].to(dev);initial=tg[:,:8,0]
    with torch.no_grad(): parent=model.forward_anchored(bs,cd,initial_state=initial)
    fam=row['family'];residuals[fam].append((tg-parent).cpu());targets[fam].append(tg.cpu())
    si=row['source_index'];vel=torch.from_numpy(np.asarray(src['velocity_mps'][si],np.float32))[None,None].to(dev);f0=torch.tensor([float(src['source_f0_hz'][si])],device=dev);t0=torch.tensor([float(src['source_t0_s'][si])],device=dev)
    for e in ('E0','E1','E2'): features[e][fam].append(summary_encoding_features(e,cd,vel,f0,t0,model=model,initial_state=initial).cpu())
  bundle={'schema':'b2_v6_residual_pod_bundle_v1','rank':a.rank,'families':{},'parent_checkpoint_sha256':sha(a.checkpoint)};report={'schema':'b2_v6_offline_pretrain_report_v1','families':{}}
  for fam in FAMILIES:
   r=torch.cat(residuals[fam]);t=torch.cat(targets[fam]);modes,eigs=fit_residual_pod(r,a.rank);coeff=pod_coefficients(r,modes);recon=torch.einsum('nr,rtczx->ntczx',coeff,modes)
   base_err=(r[:,8:].reshape(len(r),-1).norm(dim=1)/t[:,8:].reshape(len(t),-1).norm(dim=1).clamp_min(1e-16));new_err=((r-recon)[:,8:].reshape(len(r),-1).norm(dim=1)/t[:,8:].reshape(len(t),-1).norm(dim=1).clamp_min(1e-16));gain=(base_err-new_err)/base_err.clamp_min(1e-16)
   priors={e:fit_ridge_prior(torch.cat(features[e][fam]),coeff) for e in ('E0','E1','E2')}
   bundle['families'][fam]={'modes':modes.half(),'eigenvalues':eigs,'priors':priors}
   report['families'][fam]={'records':len(r),'mean_oracle_gain':float(gain.mean()),'minimum_oracle_gain':float(gain.min()),'mean_parent_rel':float(base_err.mean()),'mean_projected_rel':float(new_err.mean())}
  report['passed']=all(report['families'][f]['mean_oracle_gain']>=pre['gates']['minimum_mean_oracle_gain_per_family'] for f in FAMILIES)
  torch.save(bundle,out/'pod_bundle.pt');report['bundle_sha256']=sha(out/'pod_bundle.pt');atom(report,out/'report.json');atom({'status':'passed' if report['passed'] else 'rejected','report':str(out/'report.json')},terminal);print(json.dumps(report,indent=2))
 except Exception as e:
  import traceback;atom({'status':'failed','error':repr(e),'traceback':traceback.format_exc()},terminal);raise
if __name__=='__main__':main()
