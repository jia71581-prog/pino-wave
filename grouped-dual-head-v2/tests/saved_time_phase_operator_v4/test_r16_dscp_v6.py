from __future__ import annotations
from types import SimpleNamespace
import pytest,torch

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AccessLedger,Lineage
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v5 import V5ProductionBackend
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v6 import *
from scripts.train_r16_dscp_v6 import V6StageDispatcher,run_identity

def key(sample="s",parent="p"):return V6CacheKey("train",sample,"g","n",parent,"b","f")
def entry(k=None):
 k=k or key();t=torch.ones(2,3);return V6CacheEntry(k,None,t,t,(t,),t,0,1.,t,tensor_bytes((t,t,(t,),t,t)))

def test_cache_key_drift_hits_misses_and_no_serialization(tmp_path):
 cache=V6RecordCache("smoke",available=lambda:10**12);cache.put(entry());ledger=AccessLedger("train");assert cache.get(key(),ledger).key.sample_id=="s" and ledger.events[-1]["event"]=="cache_hit"
 with pytest.raises(CacheBindingRefusal):cache.get(key(parent="drift"),ledger)
 assert cache.payload()["misses"]==1 and cache.payload()["hits"]==1 and cache.payload()["serialization"] is False and list(tmp_path.iterdir())==[]

def test_cache_memory_gate_capacity_and_tensor_estimator():
 t=torch.zeros(10,dtype=torch.float32);assert tensor_bytes({"x":t})==40
 cache=V6RecordCache("smoke",available=lambda:HOST_RESERVE_BYTES+39)
 with pytest.raises(CacheMemoryRefusal):cache.put(V6CacheEntry(key(),None,t,t,(),t,0,1.,t,40))
 cache=V6RecordCache("smoke",available=lambda:10**12)
 for i in range(3):cache.put(entry(key(str(i))))
 with pytest.raises(CacheMemoryRefusal):cache.put(entry(key("4")))
 gate=stage_memory_gate("long",100,HOST_RESERVE_BYTES+21600);assert gate["passed"] and gate["cache_bytes"]==21600
 assert not stage_memory_gate("long",100,HOST_RESERVE_BYTES+21599)["passed"]

class FakeCache:
 def __init__(self):self.hits=0;self.misses=0;self.peak=0
 def payload(self):return {"hits":self.hits,"misses":self.misses,"peak_items":self.peak,"peak_bytes":123,"serialization":False}
 def clear(self):self.cleared=True
class FakeSmokeBackend:
 def __init__(self):self.cache=FakeCache();self.parents=0;self.updates=0;self.lineage=SimpleNamespace(run_digest="r",input_checkpoint_sha256="none");self.parent_binding="p";self.basis_binding="b";self.feature_binding="f";self.protocol=SimpleNamespace(payload=lambda:{"finite":True});self.peak_bytes=1
 def preload(self,i):self.parents+=1;self.cache.misses+=1;self.cache.peak+=1
 def update_cached(self,i):self.updates+=1;self.cache.hits+=1;return {"loss":1. if self.updates==1 else .1}
 def score_cached(self,i):self.cache.hits+=1;family=("uniform","layered","marmousi")[i];return {"sample_id":str(i),"ledger_digest":str(i),"family":family,"mean_frame_rel_l2":.1,"parent_mean_frame_rel_l2":.4,"aggregate_rel_l2":.1,"parent_rel_l2":.4,"finite":True}

def test_smoke_parent_exactly_three_cache_hits_and_accuracy_gate():
 backend=FakeSmokeBackend();terminal=[];result=run_v6_smoke(backend=backend,records=[0,1,2],quick_gate=lambda rows:False,checkpoint=lambda:{"best":"b"},resource_snapshot=lambda s:{"peak_bytes":1,"checkpoint_bytes":1,"space_passed":True},terminal=terminal.append,clock=lambda:0.)
 assert result["status"]=="passed" and result["updates"]==192 and backend.parents==3 and result["cache"]["misses"]==3 and result["cache"]["hits"]>192 and backend.cache.cleared

def test_truth_order_then_cached_updates_do_not_add_misses(monkeypatch):
 backend=object.__new__(V6ProductionBackend);backend.cache=V6RecordCache("smoke",available=lambda:10**12);backend.parent_binding="p";backend.basis_binding="b";backend.feature_binding="f";backend.nontruth_by_sample={"s":"n"};backend.index_keys={};backend.split="train"
 public=SimpleNamespace(sample_id="s",group_id="g",input_digest="n",velocity_mps=torch.ones(1),source_parameters=torch.ones(1),source_map=torch.ones(1),observed_wavefield=torch.ones(1));ledger=AccessLedger("train");ledger.add("candidate_sealed",serialized=True);prepared=SimpleNamespace(public=public,parent=torch.ones(1),travel=torch.ones(1),args=(torch.ones(1),),features=torch.ones(1),route_index=0,condition=1.,ledger=ledger)
 monkeypatch.setattr(V5ProductionBackend,"prepare",lambda self,index,suffix:prepared);monkeypatch.setattr(V5ProductionBackend,"open_truth",lambda self,p:p.ledger.add("future_truth_read") or torch.ones(1));monkeypatch.setattr(V6ProductionBackend,"cleanup",lambda self,p:None)
 cached=backend.preload(0);assert [e["event"] for e in ledger.events].index("candidate_sealed")<[e["event"] for e in ledger.events].index("future_truth_read") and backend.cache.payload()["misses"]==1
 lookup=AccessLedger("train");backend.cache.get(cached.key,lookup);backend.cache.get(cached.key,lookup);assert backend.cache.payload()["hits"]==2 and backend.cache.payload()["misses"]==1

def test_pilot_long_capacity_reuse_and_streaming_eval_release():
 pilot=V6RecordCache("pilot",available=lambda:10**12)
 for i in range(48):pilot.put(entry(key(str(i))))
 ledger=AccessLedger("train")
 for _ in range(3):pilot.get(key("0"),ledger)
 assert pilot.peak_items==48 and pilot.misses==48 and pilot.hits==3
 long=V6RecordCache("long",available=lambda:10**12)
 for i in range(216):long.put(entry(key(str(i))))
 assert long.peak_items==216
 evaluation=V6RecordCache("validation-once",available=lambda:10**12);e=entry();evaluation.put(e);evaluation.release(e.key);assert evaluation.current_bytes==0 and not evaluation.entries

def test_complete_run_identity_has_resume_fields():
 identity=complete_run_identity({"candidate":"r16_dscp_v6"},authorization_sha256="a",engine_sha256="e",script_sha256="s",config_sha256="c",panels_sha256="p",basis_sha256="b",parent_sha256="par",input_checkpoint_sha256="i")
 for keyname in ("script_sha256","panels_sha256_effective","basis_sha256_effective","parent_sha256_effective","engine_sha256","config_sha256_effective","input_checkpoint_sha256"):assert keyname in identity

def test_smoke_terminal_full_lineage_and_dispatch_refuses_v5():
 backend=FakeSmokeBackend();terminal=[];result=run_v6_smoke(backend=backend,records=[0,1,2],quick_gate=lambda rows:True,checkpoint=lambda:{"best":"b","last":"l"},resource_snapshot=lambda s:{"peak_bytes":1,"checkpoint_bytes":1,"space_passed":True},terminal=terminal.append,clock=lambda:0.)
 for field in ("schema","candidate","mode","status","decision","run_digest","effective","input_checkpoint_sha256","checkpoint","cache","latency","peak_vram_bytes","gates"):assert field in result
 with pytest.raises(TypeError):V6StageDispatcher(object())

def test_v6_config_and_production_source_cache_only():
 import inspect,yaml,scripts.train_r16_dscp_v6 as cli
 source=inspect.getsource(cli);assert "build_v6_backend" in source and "V5ProductionBackend" not in source and "complete_unpromoted" not in source
 config=yaml.safe_load((cli.ROOT/"configs/r16_dscp_v6.yaml").read_text());assert config["cache"]["roles"]["long"]==216 and config["cache"]["host_reserve_gib"]==16 and config["cache"]["smoke_parent_callback_maximum"]==3
 for name,command in config["commands"].items():
  if name not in {"prep","authorize","scale_decide"}:assert command.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=")
 source=inspect.getsource(cli);assert "build_v6_backend" in source and "V5ProductionBackend" not in source and "complete_unpromoted" not in source and "dispatcher.preload(train+cal)" in source
