#!/usr/bin/env python3
"""Group-disjoint online self-supervised POD adaptation for E0/E1/E2 priors."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from collections import Counter
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
for v in (str(ROOT),str(ROOT/'src')):
    if v not in sys.path:sys.path.insert(0,v)
from saved_time_phase_operator_v4.instance_adaptation.b2_v6_pod import PODAdaptConfig,adapt_pod_coefficients,predict_ridge_prior,summary_encoding_features
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator
ARMS=('zero','E0','E1','E2');FAMILIES=('uniform','layered','marmousi')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def atom(x,p):
 q=p.with_name(f'{p.name}.partial.{os.getpid()}');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def rel(a,b):return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-16))
def main():
 p=argparse.ArgumentParser();
 for n in ('checkpoint','cache','manifest','bundle','preregistration','output_dir'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
 a=p.parse_args();out=a.output_dir
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);(out/'candidates').mkdir();terminal=out/'terminal.json'
 try:
  pre=json.load(open(a.preregistration));b=pre['bindings']
  for path,key in ((Path(__file__),'evaluator_sha256'),(a.checkpoint,'checkpoint_sha256'),(a.cache,'holdout_cache_sha256'),(a.manifest,'holdout_manifest_sha256'),(a.bundle,'bundle_sha256')):
   if sha(path)!=b[key]:raise RuntimeError(f'binding drift {path}')
  m=json.load(open(a.manifest));pod=torch.load(a.bundle,map_location='cpu',weights_only=False);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);ident=ck['identity'];dev=torch.device('cuda')
  model=FrameConditionedPropagator(state_channels=8,cond_channels=7,width=ident['width'],spectral_rank=ident['spectral_rank'],modes=24,depth=4,gate_init=1.,activation_checkpointing=False).to(dev);model.load_state_dict(ck['model_state']);model.eval();records=[]
  for i,row in enumerate(m['records']):
   with h5py.File(a.cache,'r',swmr=True) as h:
    bs=torch.from_numpy(h['base_seq'][i:i+1].astype(np.float32))[:,:,None].to(dev);cd=torch.from_numpy(h['cond'][i:i+1].astype(np.float32)).to(dev);obs=torch.from_numpy(h['target'][i:i+1,:8].astype(np.float32))[:,:,None].to(dev)
   with h5py.File(m['source_h5'],'r',swmr=True) as src:
    si=row['source_index'];vel=torch.from_numpy(np.asarray(src['velocity_mps'][si],np.float32))[None].to(dev);f0=torch.tensor([float(src['source_f0_hz'][si])],device=dev);t0=torch.tensor([float(src['source_t0_s'][si])],device=dev);time=np.asarray(src['time_s'][:],float);x=np.asarray(src['x_m'][:],float);z=np.asarray(src['z_m'][:],float)
   with torch.no_grad():parent=model.forward_anchored(bs,cd,initial_state=obs[:,:,0])
   wt=time[row['window_start']:row['window_start']+64];off=max(8,min(int(np.searchsorted(wt,float(t0)+1.5/float(f0))),61));family_bundle=pod['families'][row['family']];modes=family_bundle['modes'].float().to(dev)
   features={e:summary_encoding_features(e,cd,vel[:,None],f0,t0,model=model,initial_state=obs[:,:,0]) for e in ('E0','E1','E2')};sealed={};adapt={}
   for arm in ARMS:
    prior=torch.zeros(pod['rank'],device=dev) if arm=='zero' else predict_ridge_prior(family_bundle['priors'][arm],features[arm])[0]
    cand,rep=adapt_pod_coefficients(parent,modes,prior,obs,vel,source_off_frame=off,dt_s=float(np.median(np.diff(time))),dx_m=float(np.median(np.diff(x))),dz_m=float(np.median(np.diff(z))),config=PODAdaptConfig())
    path=out/'candidates'/f'{i:02d}_{row["sample_id"]}_{arm}.pt';torch.save({'candidate':cand.cpu().half(),'report':rep},path);sealed[arm]=(path,sha(path));adapt[arm]=rep
   with h5py.File(a.cache,'r',swmr=True) as h:future=torch.from_numpy(h['target'][i:i+1,8:].astype(np.float32))[:,:,None]
   parent_cpu=parent.cpu();pr=rel(parent_cpu[:,8:],future);rr={'sample_id':row['sample_id'],'family':row['family'],'parent':pr,'source_off':off,'arms':{}}
   for arm in ARMS:
    cand=torch.load(sealed[arm][0],map_location='cpu',weights_only=False)['candidate'].float();v=rel(cand[:,8:],future);rr['arms'][arm]={'relative_l2':v,'gain':(pr-v)/max(pr,1e-16),'sha256':sealed[arm][1],'adaptation':adapt[arm]}
   records.append(rr);del bs,cd,obs,vel,parent;torch.cuda.empty_cache()
  summary={'schema':'b2_v6_online_pod_panel_v1','records':records,'arms':{},'validation_opened':False,'test_id_opened':False}
  for arm in ARMS:
   vals=np.array([r['arms'][arm]['relative_l2'] for r in records]);parents=np.array([r['parent'] for r in records]);summary['arms'][arm]={'aggregate':float(vals.mean()),'parent_aggregate':float(parents.mean()),'relative_gain':float((parents.mean()-vals.mean())/parents.mean()),'nonworse':int((vals<=parents).sum()),'mean_runtime_s':float(np.mean([r['arms'][arm]['adaptation']['elapsed_s'] for r in records])),'per_family':{f:float(np.mean([r['arms'][arm]['relative_l2'] for r in records if r['family']==f])) for f in FAMILIES}}
  primary=summary['arms']['E2'];zero=summary['arms']['zero'];g=pre['gates'];summary['judgement']={'gain_pass':primary['relative_gain']>=g['minimum_relative_gain'],'nonworse_pass':primary['nonworse']>=g['minimum_nonworse'],'runtime_pass':primary['mean_runtime_s']<=g['maximum_mean_runtime_s'],'encoding_increment_pass':zero['aggregate']-primary['aggregate']>=g['minimum_absolute_encoding_increment'],'family_pass':all(primary['per_family'][f]<summary['arms']['zero']['per_family'][f] for f in FAMILIES)};summary['judgement']['passed']=all(summary['judgement'].values());atom(summary,out/'summary.json');atom({'status':'passed' if summary['judgement']['passed'] else 'rejected','summary':str(out/'summary.json')},terminal);print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
 except Exception as e:
  import traceback;atom({'status':'failed','error':repr(e),'traceback':traceback.format_exc()},terminal);raise
if __name__=='__main__':main()
