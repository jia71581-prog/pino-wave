#!/usr/bin/env python3
"""Rewindow selected train records using source-only causal metadata."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import h5py, numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.b2_v5_components import causal_window_start

def atomic(payload,path):
    partial=path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(partial,path)
def canonical(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-manifest',type=Path,required=True); p.add_argument('--indices'); p.add_argument('--all',action='store_true'); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    source_manifest=json.loads(a.source_manifest.read_text())
    if a.all == (a.indices is not None): raise ValueError('provide exactly one of --all or --indices')
    indices=list(range(len(source_manifest['records']))) if a.all else [int(x) for x in a.indices.split(',')]
    selected=[source_manifest['records'][i] for i in indices]
    with h5py.File(source_manifest['source_h5'],'r',swmr=True) as h:
        time=np.asarray(h['time_s'][:],float); rows=[]
        for row in selected:
            i=int(row['source_index']); f0=float(h['source_f0_hz'][i]); t0=float(h['source_t0_s'][i]); item=dict(row)
            item.update(source_f0_hz=f0,source_t0_s=t0,window_start=causal_window_start(time,source_t0_s=t0,source_f0_hz=f0,k_frames=64))
            rows.append(item)
    body={'schema':'b2_v5_causal_stratified_manifest_v1','created_utc':datetime.now(timezone.utc).isoformat(),'source_h5':source_manifest['source_h5'],'source_h5_byte_count':Path(source_manifest['source_h5']).stat().st_size,'split':'train','families':['uniform','layered','marmousi'],'groups_per_family':len(rows)//3,'seed':0,'selection_rule':f'rewindow exact indices from {a.source_manifest}','n_visible':8,'k_frames':64,'window_rule':'searchsorted(time_s, source_t0_s - lead_cycles/source_f0_hz)','lead_cycles':0.5,'future_truth_opened_for_window_selection':False,'source_manifest':str(a.source_manifest),'source_manifest_indices':indices,'records':rows,'validation_opened':False,'test_id_opened':False}
    body['selection_sha256']=canonical(body); atomic(body,a.output); print(json.dumps({'output':str(a.output),'records':len(rows),'selection_sha256':body['selection_sha256']},indent=2))
if __name__=='__main__': main()
