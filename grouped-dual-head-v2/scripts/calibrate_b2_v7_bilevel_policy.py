#!/usr/bin/env python3
"""Offline future-truth calibration of a conservative online POD policy."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
for v in (str(ROOT),str(ROOT/'src')):
    if v not in sys.path:sys.path.insert(0,v)
from saved_time_phase_operator_v4.instance_adaptation.b2_v6_pod import PODAdaptConfig,adapt_pod_coefficients,predict_ridge_prior,summary_encoding_features
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def atom(x,p):q=p.with_name(f'{p.name}.partial.{os.getpid()}');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def rel(a,b):return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-16))
def main():
 p=argparse.ArgumentParser();
 for n in ('checkpoint','cache','manifest','bundle','preregistration','output_dir'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
 a=p.parse_args();out=a.output_dir
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);term=out/'terminal.json'
 try:
  pre=json.load(open(a.preregistration));b=pre['bindings']
  for path,key in ((Path(__file__),'calibrator_sha256'),(a.checkpoint,'checkpoint_sha256'),(a.cache,'fit_cache_sha256'),(a.manifest,'fit_manifest_sha256'),(a.bundle,'bundle_sha256')):
   if sha(path)!=b[key]:raise RuntimeError(f'binding drift {path}')
  m=json.load(open(a.manifest));pod=torch.load(a.bundle,map_location='cpu',weights_only=False);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);ident=ck['identity'];dev=torch.device('cuda')
  with h5py.File(a.cache,'r',swmr=True) as h:base=torch.from_numpy(h['base_seq'][:].astype(np.float32))[:,:,None];target=torch.from_numpy(h['target'][:].astype(np.float32))[:,:,None];cond=torch.from_numpy(h['cond'][:].astype(np.float32))
  model=FrameConditionedPropagator(state_channels=8,cond_channels=7,width=ident['width'],spectral_rank=ident['spectral_rank'],modes=24,depth=4,gate_init=1.,activation_checkpointing=False).to(dev);model.load_state_dict(ck['model_state']);model.eval();records=[]
  with h5py.File(m['source_h5'],'r',swmr=True) as src:
   for i,row in enumerate(m['records']):
    bs=base[i:i+1].to(dev);cd=cond[i:i+1].to(dev);tg=target[i:i+1].to(dev);obs=tg[:,:8];si=row['source_index'];vel=torch.from_numpy(np.asarray(src['velocity_mps'][si],np.float32))[None].to(dev);f0=torch.tensor([float(src['source_f0_hz'][si])],device=dev);t0=torch.tensor([float(src['source_t0_s'][si])],device=dev);time=np.asarray(src['time_s'][:],float);x=np.asarray(src['x_m'][:],float);z=np.asarray(src['z_m'][:],float)
    with torch.no_grad():parent=model.forward_anchored(bs,cd,initial_state=obs[:,:,0])
    wt=time[row['window_start']:row['window_start']+64];off=max(8,min(int(np.searchsorted(wt,float(t0)+1.5/float(f0))),61));feat=summary_encoding_features('E2',cd,vel[:,None],f0,t0,model=model,initial_state=obs[:,:,0]);prior=predict_ridge_prior(pod['families'][row['family']]['priors']['E2'],feat)[0];records.append({'family':row['family'],'parent':parent.detach(),'target':tg.detach(),'obs':obs.detach(),'vel':vel.detach(),'modes':pod['families'][row['family']]['modes'].float().to(dev),'prior':prior.detach(),'off':off,'dt':float(np.median(np.diff(time))),'dx':float(np.median(np.diff(x))),'dz':float(np.median(np.diff(z)))})
  grid=[]
  for ps in (0.,.02,.05,.1):
   for ww in (0.,.01,.05):
    for lr in (.01,.03):
     candidates=[];parents=[];targets=[]
     for r in records:
      c,_=adapt_pod_coefficients(r['parent'],r['modes'],ps*r['prior'],r['obs'],r['vel'],source_off_frame=r['off'],dt_s=r['dt'],dx_m=r['dx'],dz_m=r['dz'],config=PODAdaptConfig(weak_weight=ww,steps=4,learning_rate=lr));candidates.append(c);parents.append(r['parent']);targets.append(r['target'])
     for alpha in (.05,.1,.25,.5,1.):
      vals=[]
      for c,pn,tg in zip(candidates,parents,targets):vals.append(rel((pn+alpha*(c-pn))[:,8:],tg[:,8:]))
      grid.append({'prior_scale':ps,'weak_weight':ww,'learning_rate':lr,'steps':4,'output_scale':alpha,'aggregate':float(np.mean(vals))})
  best=min(grid,key=lambda r:r['aggregate']);parent_agg=float(np.mean([rel(r['parent'][:,8:],r['target'][:,8:]) for r in records]));best['parent_aggregate']=parent_agg;best['improved']=best['aggregate']<parent_agg
  atom({'schema':'b2_v7_bilevel_policy_v1','policy':best,'grid_size':len(grid)},out/'policy.json');atom({'status':'complete','improved':best['improved'],'policy':str(out/'policy.json')},term);print(json.dumps(best,indent=2))
 except Exception as e:
  import traceback;atom({'status':'failed','error':repr(e),'traceback':traceback.format_exc()},term);raise
if __name__=='__main__':main()
