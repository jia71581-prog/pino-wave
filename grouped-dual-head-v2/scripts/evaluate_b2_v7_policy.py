#!/usr/bin/env python3
"""Evaluate one frozen bilevel-calibrated online policy on fresh train groups."""
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
 for n in ('checkpoint','cache','manifest','bundle','policy','preregistration','output_dir'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
 a=p.parse_args();out=a.output_dir
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);(out/'candidates').mkdir();term=out/'terminal.json'
 try:
  pre=json.load(open(a.preregistration));b=pre['bindings']
  for path,key in ((Path(__file__),'evaluator_sha256'),(a.checkpoint,'checkpoint_sha256'),(a.cache,'holdout_cache_sha256'),(a.manifest,'holdout_manifest_sha256'),(a.bundle,'bundle_sha256'),(a.policy,'policy_sha256')):
   if sha(path)!=b[key]:raise RuntimeError(f'binding drift {path}')
  m=json.load(open(a.manifest));pod=torch.load(a.bundle,map_location='cpu',weights_only=False);policy=json.load(open(a.policy))['policy'];ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);ident=ck['identity'];dev=torch.device('cuda');model=FrameConditionedPropagator(state_channels=8,cond_channels=7,width=ident['width'],spectral_rank=ident['spectral_rank'],modes=24,depth=4,gate_init=1.,activation_checkpointing=False).to(dev);model.load_state_dict(ck['model_state']);model.eval();rows=[]
  for i,row in enumerate(m['records']):
   with h5py.File(a.cache,'r',swmr=True) as h:bs=torch.from_numpy(h['base_seq'][i:i+1].astype(np.float32))[:,:,None].to(dev);cd=torch.from_numpy(h['cond'][i:i+1].astype(np.float32)).to(dev);obs=torch.from_numpy(h['target'][i:i+1,:8].astype(np.float32))[:,:,None].to(dev)
   with h5py.File(m['source_h5'],'r',swmr=True) as src:si=row['source_index'];vel=torch.from_numpy(np.asarray(src['velocity_mps'][si],np.float32))[None].to(dev);f0=torch.tensor([float(src['source_f0_hz'][si])],device=dev);t0=torch.tensor([float(src['source_t0_s'][si])],device=dev);time=np.asarray(src['time_s'][:],float);x=np.asarray(src['x_m'][:],float);z=np.asarray(src['z_m'][:],float)
   with torch.no_grad():parent=model.forward_anchored(bs,cd,initial_state=obs[:,:,0])
   feat=summary_encoding_features('E2',cd,vel[:,None],f0,t0,model=model,initial_state=obs[:,:,0]);prior=predict_ridge_prior(pod['families'][row['family']]['priors']['E2'],feat)[0]*policy['prior_scale'];wt=time[row['window_start']:row['window_start']+64];off=max(8,min(int(np.searchsorted(wt,float(t0)+1.5/float(f0))),61));cand,rep=adapt_pod_coefficients(parent,pod['families'][row['family']]['modes'].float().to(dev),prior,obs,vel,source_off_frame=off,dt_s=float(np.median(np.diff(time))),dx_m=float(np.median(np.diff(x))),dz_m=float(np.median(np.diff(z))),config=PODAdaptConfig(weak_weight=policy['weak_weight'],steps=policy['steps'],learning_rate=policy['learning_rate']));cand=parent+policy['output_scale']*(cand-parent);path=out/'candidates'/f'{i:02d}_{row["sample_id"]}.pt';torch.save({'candidate':cand.cpu().half(),'adaptation':rep},path);digest=sha(path)
   with h5py.File(a.cache,'r',swmr=True) as h:future=torch.from_numpy(h['target'][i:i+1,8:].astype(np.float32))[:,:,None]
   sealed=torch.load(path,map_location='cpu',weights_only=False)['candidate'].float();pr=rel(parent.cpu()[:,8:],future);ar=rel(sealed[:,8:],future);rows.append({'sample_id':row['sample_id'],'family':row['family'],'parent':pr,'adapted':ar,'gain':(pr-ar)/pr,'sha256':digest,'adaptation':rep})
  pv=np.array([r['parent'] for r in rows]);av=np.array([r['adapted'] for r in rows]);summary={'schema':'b2_v7_policy_holdout_v1','parent_aggregate':float(pv.mean()),'adapted_aggregate':float(av.mean()),'relative_gain':float((pv.mean()-av.mean())/pv.mean()),'nonworse':int((av<=pv).sum()),'per_family':{f:{'parent':float(np.mean([r['parent'] for r in rows if r['family']==f])),'adapted':float(np.mean([r['adapted'] for r in rows if r['family']==f]))} for f in ('uniform','layered','marmousi')},'records':rows,'validation_opened':False,'test_id_opened':False};summary['passed']=summary['adapted_aggregate']<summary['parent_aggregate'];atom(summary,out/'summary.json');atom({'status':'passed' if summary['passed'] else 'rejected','summary':str(out/'summary.json')},term);print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
 except Exception as e:
  import traceback;atom({'status':'failed','error':repr(e),'traceback':traceback.format_exc()},term);raise
if __name__=='__main__':main()
