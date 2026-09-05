from __future__ import annotations
import copy, json, random
import h5py, numpy as np, pytest, torch
from pathlib import Path
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import BindingRefusal,checkpoint_payload,load_checkpoint,make_optimizer,save_best_last
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import *
from saved_time_phase_operator_v4.instance_adaptation.contracts import SnapshotAccessAudit

def basis():
 g=torch.Generator().manual_seed(372);out=[]
 for _ in range(3):out.append(torch.linalg.qr(torch.randn(401,16,generator=g,dtype=torch.float64))[0].float())
 return torch.stack(out)

def test_actual_model_bf16_forward_backward_optimizer():
 m=R16DSCP(basis(),torch.ones(3,16));o=make_optimizer(list(m.parameters()));x=torch.randn(1,29,9,11).bfloat16();o.zero_grad();
 with torch.autocast("cpu",dtype=torch.bfloat16): loss=m.coefficient_head(x).float().square().mean()
 loss.backward();o.step();assert torch.isfinite(loss)

def test_actual_model_resume_equivalence(tmp_path):
 random.seed(372);np.random.seed(372);torch.manual_seed(372); b=basis();m=R16DSCP(b,torch.ones(3,16));initial=copy.deepcopy(m.state_dict());o=make_optimizer(list(m.parameters()))
 def step(mm,oo):
  x=torch.randn(1,29,5,7);oo.zero_grad();z=mm.coefficient_head(x).square().mean();z.backward();oo.step()
 for _ in range(4):step(m,o)
 random.seed(372);np.random.seed(372);torch.manual_seed(372);a=R16DSCP(b,torch.ones(3,16));a.load_state_dict(initial);ao=make_optimizer(list(a.parameters()))
 for _ in range(2):step(a,ao)
 ident={"candidate":"r16_dscp_v3","run":"x"};p=checkpoint_payload(a,ao,run_identity=ident,sampler_order=[1],progress={"u":2});paths=save_best_last(p,tmp_path,is_best=True)
 r=R16DSCP(b,torch.ones(3,16));ro=make_optimizer(list(r.parameters()));load_checkpoint(paths["last"],r,ro,expected_run_identity=ident)
 for _ in range(2):step(r,ro)
 assert all(torch.equal(x,y) for x,y in zip(m.parameters(),r.parameters()))

def test_feature_truth_mutation_invariance():
 m=R16DSCP(basis(),torch.ones(3,16));x=torch.randn(1,29,7,7);truth=torch.randn(4,7,7);a=m.coefficient_head(x).detach();truth.mul_(999);assert torch.equal(a,m.coefficient_head(x).detach())

def test_access_order_exact_observations_and_seal(tmp_path):
 p=tmp_path/"x.h5"
 with h5py.File(p,"w") as h:h.create_dataset("wavefield",data=np.arange(1*8*2*2,dtype=np.float32).reshape(1,8,2,2))
 l=AccessLedger("train");obs=read_exact_observations(p,0,2,3,split="train",ledger=l);assert obs.shape[0]==2
 seal_candidate(torch.zeros(4,2,2),l);line,auth=_authorization("smoke");future=authorized_future_truth(p,0,3,mode="smoke",split="train",ledger=l,authorization=auth,expected_lineage=line);assert future.shape[0]==4
 assert [e["index"] for e in l.events if e["event"]=="observed_truth_read"]==[2,3];l.require_before("candidate_sealed","future_truth_read")

def test_nontrain_refusal_for_train_modes():
 with pytest.raises(LeakageRefusal):assert_stage_split("pilot","validation")
 assert_stage_split("validation-once","validation")

def test_authorization_hash_drift_refusal():
 line=Lineage("r16_dscp_v3","smoke","r","c","cfg","p","b","bt","par","in")
 base={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"gates_sha256":"g"};base["authorization_digest"]=canonical_sha(base);validate_authorization(base,line)
 base["code_sha256"]="bad"
 with pytest.raises(Exception):validate_authorization(base,line)

def _authorization(mode="validation-once"):
 line=Lineage("r16_dscp_v3",mode,"r","c","cfg","p","b","bt","par","in");a={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"gates_sha256":"g"};a["authorization_digest"]=canonical_sha(a);return line,a

def test_once_token_only_immediately_before_truth(tmp_path):
 p=tmp_path/"x.h5";token=tmp_path/"token.json"
 with h5py.File(p,"w") as h:h.create_dataset("wavefield",data=np.zeros((1,5,1,1),np.float32))
 l=AccessLedger("validation");line,auth=_authorization()
 with pytest.raises(Exception):authorized_future_truth(p,0,1,mode="validation-once",split="validation",ledger=l,authorization=auth,expected_lineage=line,once_token=token)
 assert not token.exists();seal_candidate(torch.zeros(3,1,1),l);authorized_future_truth(p,0,1,mode="validation-once",split="validation",ledger=l,authorization=auth,expected_lineage=line,once_token=token);assert token.exists();inode=token.stat().st_ino
 l2=AccessLedger("validation");seal_candidate(torch.zeros(3,1,1),l2);authorized_future_truth(p,0,1,mode="validation-once",split="validation",ledger=l2,authorization=auth,expected_lineage=line,once_token=token);assert token.stat().st_ino==inode and any(e["event"]=="once_token_already_claimed" for e in l2.events)

def test_guarded_public_loader_validation_exposes_exact_two_frames():
 audit=SnapshotAccessAudit((2,3));audit.read((2,3))
 record=type("R",(),dict(velocity_mps=torch.ones(2,2),source_parameters=torch.ones(5),source_map=torch.ones(2,2),time_s=torch.arange(5),x_m=torch.arange(2),z_m=torch.arange(2),observed_indices=(2,3),observed_wavefield=torch.ones(2,2,2),dense_travel_time_s=None,input_digest="d",source_index=7,sample_id="s",group_id="g",audit=audit))()
 class D:
  def __init__(self,*a,**k):self.record=record
  def __len__(self):return 1
  def __getitem__(self,i):return self.record
  def close(self):pass
 loader=GuardedPublicLoader("unused",object(),split="validation",dataset_factory=D);bundle=loader[0]
 assert bundle.observed_indices==(2,3) and bundle.observed_wavefield.shape[0]==2

def test_dispatch_update_eval_paths_are_separate():
 assert [p.action for p in stage_dispatch("pilot")]==["update","ridge_fit","eval"]
 assert all(p.action=="eval" for p in stage_dispatch("final-train-confirm"))
 assert stage_dispatch("validation-once")[0].split=="validation"
 assert stage_dispatch("scale-probe-4")[0].action=="ephemeral_update_ddp"
 with pytest.raises(LeakageRefusal):assert_stage_split("test-once","validation")

def test_metrics_gate_ridge_and_ddp():
 t=torch.ones(6,4,4);p=t*1.2;c=t*1.05;r=metric_record(c,p,t,family="uniform",condition=1,abstain=False);rows=[{**r,"sample_id":str(i)} for i in range(24)];a=aggregate_metrics(rows);assert a["nonworse"]==24
 assert ddp_host_aggregate([rows[:12],rows[12:]])["digest"]==ddp_host_aggregate([rows[12:],rows[:12]])["digest"]
 ridge=StreamingRidge();x=torch.randn(1,29,2,2);y=torch.randn(1,16,2,2);ridge.update(x,y);assert ridge.solve().weight.shape==(16,29,1,1)

def test_scale_math():
 one={"wall_s":4.,"metric_digest":"m","model_sha256":"h","data_sha256":"d","schedule_sha256":"s"};four={"wall_s":1.,"metric_digest":"m","model_sha256":"h","data_sha256":"d","schedule_sha256":"s","per_gpu_peak_bytes":[1]*4};assert scale_decision(one,four)["selected_gpus"]==4

def _score_row(family="uniform",candidate=.1,parent=.3):
 return {"sample_id":family,"family":family,"aggregate_rel_l2":candidate,"parent_rel_l2":parent,"mean_frame_rel_l2":candidate,"parent_mean_frame_rel_l2":parent,"nonworse":candidate<=parent,"time_bands":{"late":candidate},"parent_time_bands":{"late":parent},"spectrum_bands":{"high":candidate},"parent_spectrum_bands":{"high":parent},"correction_energy_ratio":.1,"finite":True}

def test_smoke_stage_end_to_end_pass_and_failure(tmp_path):
 line,_=_authorization("smoke");calls=[]
 def update(record):calls.append(record);return {"loss":1.0 if len(calls)==1 else .1}
 resources=lambda saved:{"peak_bytes":1,"checkpoint_bytes":1,"space_passed":True}
 terminal=run_smoke_stage(records=["u","l","m"],update=update,score=lambda:[_score_row(f,.1,.4) for f in ("uniform","layered","marmousi")],checkpoint=lambda:{"sha256":"c"},lineage=line,run_dir=tmp_path/"pass",max_updates=4,resource_snapshot=resources)
 assert terminal["status"]=="passed" and terminal["updates"]==4 and terminal["promotion_authorization_created"] is False
 bad=run_smoke_stage(records=[1],update=lambda r:{"loss":1.},score=lambda:[_score_row(f,.5,.4) for f in ("uniform","layered","marmousi")],checkpoint=lambda:{},lineage=line,run_dir=tmp_path/"fail",max_updates=2,resource_snapshot=resources)
 assert bad["status"]=="fail_gate" and bad["promotion_authorization_created"] is False

def test_pilot_candidate_ridge_24_confirm_gate_terminal(tmp_path):
 line,_=_authorization("pilot");fit=list(range(24));confirm=list(range(24));ridge_seen=[];updates=[]
 families=("uniform","layered","marmousi")
 candidate=[_score_row(families[i%3],.1,.3)|{"sample_id":str(i)} for i in confirm]
 ridge=[_score_row(families[i%3],.12,.3)|{"sample_id":str(i)} for i in confirm]
 terminal=run_pilot_stage(fit_records=fit,confirm_records=confirm,candidate_update=lambda r:updates.append(r) or {"loss":.1},ridge_update=ridge_seen.append,ridge_solve=lambda:"ridge",candidate_score=lambda rows:candidate,ridge_score=lambda model,rows:ridge,checkpoint=lambda:{"sha256":"c"},lineage=line,run_dir=tmp_path,max_updates=25,resource_snapshot=lambda s:{"adapter_mean_s":.1,"adapter_p95_s":.2,"e2e_mean_ratio":1.,"e2e_p95_ratio":1.,"peak_bytes":1,"checkpoint_bytes":1,"space_passed":True})
 assert len(ridge_seen)==24 and len(updates)==25 and terminal["status"]=="passed" and terminal["promotion_authorization_created"] is False

def test_prepare_record_seals_before_truth_authorization():
 audit=SnapshotAccessAudit((1,2));audit.read((1,2));public=PublicBundle(torch.ones(2,2),torch.ones(5),torch.ones(2,2),torch.arange(4),torch.arange(2),torch.arange(2),(1,2),torch.ones(2,2,2),None,"d",0,"s","g")
 prepared=prepare_public_record(public,parent_predictor=lambda p:torch.zeros(4,2,2),deployment_builder=lambda p,parent:(parent,),candidate_predictor=lambda args:args[0],split="train")
 assert prepared.ledger.events[-1]["event"]=="candidate_sealed"

def test_production_backend_callbacks_execute_without_field_artifacts(tmp_path,monkeypatch):
 import saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 as e
 public=PublicBundle(torch.ones(2,2)*2000,torch.tensor([0.,0.,20.,.1,1.]),torch.ones(2,2),torch.linspace(0,1,401),torch.arange(2),torch.arange(2),(1,2),torch.zeros(2,2,2),None,"digest",0,"s","g")
 class Loader:
  def __len__(self):return 1
  def __getitem__(self,i):return public
 class Fake(torch.nn.Module):
  def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.));self.register_buffer("bases",basis());self.register_buffer("coefficient_scales",torch.ones(3,16))
  def forward(self,*args):return args[7]+self.weight*0
 fake=Fake();optimizer=make_optimizer(list(fake.parameters()));line,auth=_authorization("smoke")
 h5=tmp_path/"truth.h5"
 with h5py.File(h5,"w") as h:h.create_dataset("wavefield",data=np.zeros((1,401,2,2),np.float32))
 decision=type("D",(),{"index":0,"abstain":False})()
 monkeypatch.setattr(e,"deployment_features",lambda *a:(torch.zeros(1,29,2,2),[decision],torch.tensor([1.])))
 backend=ProductionSmokePilotBackend(public_loader=Loader(),parent_predictor=lambda p:torch.zeros(401,2,2),travel_builder=lambda p:torch.zeros(2,2),candidate=fake,optimizer=optimizer,authorization=auth,lineage=line,source_h5=h5,run_dir=tmp_path/"run",device=torch.device("cpu"),family_by_sample={"s":"uniform"})
 prepared=backend.prepare(0);truth=backend.open_train_truth(prepared);assert prepared.ledger.events[-1]["event"]=="future_truth_read"
 score=backend.score(fake,prepared,truth);assert score["aggregate_rel_l2"]==0
 backend.ridge_update(prepared,truth);ridge=backend.ridge_finalize();rscore=backend.ridge_score(ridge,prepared,truth);assert torch.isfinite(torch.tensor(rscore["aggregate_rel_l2"]))
 saved=backend.checkpoint({"update":0});resources=backend.resources(saved);assert resources["fields_serialized"] is False and resources["coefficient_maps_serialized"] is False
 assert {p.name for p in (tmp_path/"run").iterdir()}=={"best.pt","last.pt"}

def test_cli_smoke_pilot_dispatch_new_runners_and_old_loop_unreachable(tmp_path,monkeypatch):
 import scripts.train_r16_dscp_v3 as cli
 auth=tmp_path/"auth.json";auth.write_text("{}")
 line,_=_authorization("smoke");monkeypatch.setattr(cli,"OUT",tmp_path/"out");(tmp_path/"out").mkdir();monkeypatch.setattr(cli,"require_cuda_environment",lambda **k:None)
 monkeypatch.setattr(cli,"authorize",lambda mode,path:({}, {},Lineage("r16_dscp_v3",mode,"r","c","cfg","p","b","bt","par","in")))
 calls=[];monkeypatch.setattr(cli,"_dispatch_smoke_pilot_authorized",lambda mode,authorization,line,run:calls.append((mode,run)) or {"status":"passed"})
 assert cli.run_authorized("smoke","0",auth)["status"]=="passed";assert cli.run_authorized("pilot","0",auth)["status"]=="passed";assert [x[0] for x in calls]==["smoke","pilot"]

def test_auth_failure_precedes_data_dispatch_and_backend_failure_terminal(tmp_path,monkeypatch):
 import scripts.train_r16_dscp_v3 as cli
 auth=tmp_path/"auth.json";auth.write_text("{}")
 real_dispatch=cli._dispatch_smoke_pilot_authorized
 monkeypatch.setattr(cli,"require_cuda_environment",lambda **k:None);monkeypatch.setattr(cli,"authorize",lambda *a:(_ for _ in ()).throw(BindingRefusal("drift")))
 called=[];monkeypatch.setattr(cli,"_dispatch_smoke_pilot_authorized",lambda *a,**k:called.append(1))
 with pytest.raises(BindingRefusal):cli.run_authorized("smoke","0",auth)
 assert called==[]
 run=tmp_path/"failure";run.mkdir();line,_=_authorization("smoke")
 with pytest.raises(RuntimeError):real_dispatch("smoke",{},line,run,backend_factory=lambda *a:(_ for _ in ()).throw(RuntimeError("boom")))
 terminal=json.loads((run/"terminal.json").read_text());assert terminal["status"]=="failed" and terminal["mode"]=="smoke"

def test_long_twenty_epoch_cosine_best_last_and_patience(tmp_path):
 line,_=_authorization("long");lrs=[];epochs=[]
 def evaluate(rows):
  value=.3-.01*len(epochs);return [{**_score_row(f,value,.5),"sample_id":f} for f in ("uniform","layered","marmousi")]
 def checkpoint(progress,is_best):epochs.append(progress["epoch"]);return {"size_bytes":1,"best":is_best,"last":"last.pt"}
 terminal=run_long_stage(fit_records=[1,2],calibration_records=[1],update=lambda r:{"loss":.1},evaluate=evaluate,set_lr=lrs.append,checkpoint=checkpoint,resource_snapshot=lambda s:{"space_passed":True,"peak_bytes":1},lineage=line,run_dir=tmp_path,max_epochs=20,max_seconds=10)
 assert terminal["status"]=="completed_train" and terminal["epochs_completed"]==20
 assert lrs[0]==pytest.approx(3e-3) and lrs[-1]==pytest.approx(3e-5)
 state=PatienceState();assert state.update(1,0)[1] is False
 for epoch in range(1,5):stop=state.update(1,epoch)[1]
 assert stop is True

def test_final_eval_zero_update_lock_only_on_pass_and_validation_test_contract(tmp_path):
 line,_=_authorization("final-train-confirm");rows=[{**_score_row(("uniform","layered","marmousi")[i%3],.1,.3),"sample_id":str(i)} for i in range(24)]
 resources=lambda:{"e2e_mean_ratio":1.,"e2e_p95_ratio":1.,"e2e_mean_s":1.,"e2e_p95_s":1.,"peak_bytes":1}
 terminal=run_evaluation_stage(mode="final-train-confirm",records=list(range(24)),evaluate_one=lambda i:rows[i],resource_snapshot=resources,lineage=line,run_dir=tmp_path/"final",metadata_digest="m",candidate_lock_payload={"checkpoint_sha256":"c","thresholds_sha256":"t"})
 assert terminal["status"]=="passed" and terminal["optimizer_steps"]==terminal["backward_calls"]==0 and (tmp_path/"final/candidate_lock.json").exists()
 bad=[{**row,"aggregate_rel_l2":.6,"mean_frame_rel_l2":.6} for row in rows]
 failed=run_evaluation_stage(mode="final-train-confirm",records=list(range(24)),evaluate_one=lambda i:bad[i],resource_snapshot=resources,lineage=line,run_dir=tmp_path/"bad",metadata_digest="m",candidate_lock_payload={"x":1})
 assert failed["status"]=="fail_gate" and not (tmp_path/"bad/candidate_lock.json").exists()
 vline,_=_authorization("validation-once");validation=run_evaluation_stage(mode="validation-once",records=[1,2,3],evaluate_one=lambda i:{**_score_row(("uniform","layered","marmousi")[i-1],.04,.1),"sample_id":str(i)},resource_snapshot=lambda:{**resources(),"e2e_mean_s":3.,"e2e_p95_s":3.},lineage=vline,run_dir=tmp_path/"val",metadata_digest="480digest")
 assert validation["status"]=="fail_gate"
 tline,_=_authorization("test-once")
 with pytest.raises(GateFailure):run_evaluation_stage(mode="test-once",records=[1],evaluate_one=lambda i:rows[0],resource_snapshot=resources,lineage=tline,run_dir=tmp_path/"test",metadata_digest="480test",required_validation_terminal=validation)

def test_cli_long_final_validation_test_specialized_dispatch(tmp_path,monkeypatch):
 import scripts.train_r16_dscp_v3 as cli
 auth=tmp_path/"auth.json";auth.write_text("{}");out=tmp_path/"out";out.mkdir();monkeypatch.setattr(cli,"OUT",out);monkeypatch.setattr(cli,"require_cuda_environment",lambda **k:None);monkeypatch.setattr(cli,"authorize",lambda mode,path:({}, {},Lineage("r16_dscp_v3",mode,"r","c","cfg","p","b","bt","par","in")))
 calls=[];monkeypatch.setattr(cli,"_dispatch_long_authorized",lambda *a:calls.append("long") or {"status":"completed_train"});monkeypatch.setattr(cli,"_dispatch_evaluation_authorized",lambda mode,*a:calls.append(mode) or {"status":"passed"})
 assert cli.run_authorized("long","0",auth)["status"]=="completed_train"
 for mode in ("final-train-confirm","validation-once","test-once"):assert cli.run_authorized(mode,"0",auth)["status"]=="passed"
 assert calls==["long","final-train-confirm","validation-once","test-once"]

def test_scale_one_four_aggregation_model_unchanged_failure_and_decision(tmp_path):
 records=[f"s{i:02d}" for i in range(12)];line,_=_authorization("scale-probe-1");state={"hash":"h"}
 measure=lambda record:{"sample_id":record,"loss":float(int(record[1:])+1),"ledger_digest":record}
 one=run_scale_probe(mode="scale-probe-1",records=records,measure=measure,model_hash=lambda:state["hash"],resource_snapshot=lambda:{"peak_bytes":1},lineage=line,run_dir=tmp_path/"one")
 reports=[]
 for rank in range(4):reports.append(run_scale_probe(mode="scale-probe-4",records=records,measure=measure,model_hash=lambda:"h",resource_snapshot=lambda:{"peak_bytes":1},lineage=line,run_dir=tmp_path/f"r{rank}",world_size=4,rank=rank))
 four=aggregate_scale_ranks(reports,records);assert one["metric_digest"]==four["metric_digest"] and one["model_sha256"]==four["model_sha256"]
 with pytest.raises(GateFailure):aggregate_scale_ranks(reports[:3],records)
 fast={**four,"wall_s":one["wall_s"]/4,"gpu_seconds":one["gpu_seconds"],"metric_digest":one["metric_digest"],"model_sha256":one["model_sha256"],"data_sha256":one["data_sha256"],"schedule_sha256":one["schedule_sha256"],"per_gpu_peak_bytes":[1]*4}
 assert scale_decision(one,fast)["selected_gpus"]==4
 slow={**fast,"wall_s":one["wall_s"]};assert scale_decision(one,slow)["selected_gpus"]==1

def test_scale_probe_detects_model_pollution(tmp_path):
 records=[f"s{i:02d}" for i in range(12)];state={"n":0}
 def measure(record):state["n"]+=1;return {"sample_id":record,"loss":1.,"ledger_digest":record}
 with pytest.raises(GateFailure):run_scale_probe(mode="scale-probe-1",records=records,measure=measure,model_hash=lambda:str(state["n"]),resource_snapshot=lambda:{"peak_bytes":1},lineage=_authorization("scale-probe-1")[0],run_dir=tmp_path)

def test_cli_scale_specialized_dispatch(tmp_path,monkeypatch):
 import scripts.train_r16_dscp_v3 as cli
 auth=tmp_path/"a.json";auth.write_text("{}");out=tmp_path/"out";out.mkdir();monkeypatch.setattr(cli,"OUT",out);monkeypatch.setattr(cli,"require_cuda_environment",lambda **k:None);monkeypatch.setattr(cli,"authorize",lambda mode,path:({}, {},Lineage("r16_dscp_v3",mode,"r","c","cfg","p","b","bt","par","in")))
 calls=[];monkeypatch.setattr(cli,"_dispatch_scale_probe_authorized",lambda mode,*a:calls.append(mode) or {"status":"passed"});monkeypatch.setattr(cli,"_dispatch_scale_decide_authorized",lambda *a:calls.append("scale-decide") or {"selected_gpus":1})
 assert cli.run_authorized("scale-probe-1","0",auth)["status"]=="passed";assert cli.run_authorized("scale-decide","0",auth)["selected_gpus"]==1;assert calls==["scale-probe-1","scale-decide"]

def test_authorize_stage_strict_chain_refuse_overwrite_and_no_old_loop(tmp_path,monkeypatch):
 import inspect, scripts.train_r16_dscp_v3 as cli
 out=tmp_path/"results";out.mkdir();prereg=tmp_path/"prereg.json";prereg.write_text(json.dumps({"status":"frozen"}));monkeypatch.setattr(cli,"OUT",out);monkeypatch.setattr(cli,"PREREG",prereg);monkeypatch.setattr(cli,"verify",lambda:{"run_identity":{"run_digest":"r"}})
 target=out/"smoke.json";cli.authorize_stage("smoke",target);payload=json.loads(target.read_text());assert payload["schema"]==AUTH_SCHEMA and payload["mode"]=="smoke"
 with pytest.raises(FileExistsError):cli.authorize_stage("smoke",target)
 (out/"smoke").mkdir();(out/"smoke/terminal.json").write_text(json.dumps({"status":"fail_gate"}))
 with pytest.raises(BindingRefusal):cli.authorize_stage("pilot",out/"pilot.json",tmp_path/"missing.pt")
 source=inspect.getsource(cli.run_authorized);assert "complete_unpromoted" not in source and "unhandled mode after specialized dispatch" in source
