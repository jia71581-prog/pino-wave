from __future__ import annotations
import json
from pathlib import Path
import h5py, numpy as np, pytest, torch, yaml

from saved_time_phase_operator_v4.instance_adaptation.contracts import SnapshotAccessAudit
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetRecord
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import R16DSCP
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import AUTH_SCHEMA,AccessLedger,GateFailure,Lineage,canonical_sha
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import *
from scripts.train_r16_dscp_v4 import V4StageDispatcher
from scripts.train_r16_dscp_v4 import V4LongDispatcher,launch_long_command
import scripts.train_r16_dscp_v4 as v4cli

def record(shape=(201,201)):
 audit=SnapshotAccessAudit((1,2));audit.read((1,2));h,w=shape
 return GuardedOnsetRecord(torch.ones(1,h,w)*2000,torch.tensor([10.,10.,20.,.1,1.]),torch.ones(1,h,w),torch.linspace(0,1,401),torch.arange(w),torch.arange(h),"sample","group","uniform",0,(1,2),torch.zeros(2,1,h,w),"digest",audit,None)

def model():
 q=torch.linalg.qr(torch.randn(401,16,generator=torch.Generator().manual_seed(372),dtype=torch.float64))[0].float();return R16DSCP(q.repeat(3,1,1),torch.ones(3,16))

def auth(mode="validation-once"):
 line=Lineage("r16_dscp_v3",mode,"r","c","cfg","p","b","bt","par","in");payload={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"gates_sha256":"g"};payload["authorization_digest"]=canonical_sha(payload);return line,payload

def test_canonical_guarded_shapes_and_model_args():
 public=canonical_public(record());assert public.velocity_mps.shape==(201,201) and public.source_map.shape==(201,201) and public.observed_wavefield.shape==(2,201,201)
 seen={}
 def solver(velocity,source):seen.update(velocity=velocity.shape,source=source.shape);return np.zeros_like(velocity)
 assert canonical_travel(public,solver).shape==(201,201) and seen=={"velocity":(201,201),"source":(5,)}
 velocity,source,source_map=canonical_parent_arrays(public);assert velocity.shape==source_map.shape==(201,201) and source.shape==(5,)
 args=deployment_args(public,torch.zeros(401,201,201),torch.ones(201,201),torch.device("cpu"));assert all(args[i].ndim==4 for i in (0,1,2,5,6,7));assert args[5].shape==(1,1,201,201)
 with pytest.raises(ShapeRefusal):canonical_public(record((3,4)).__class__(**{**record((3,4)).__dict__,"velocity_mps":torch.ones(2,3,4)}))

def test_observed_shape_refusal():
 r=record((3,4));object.__setattr__(r,"observed_wavefield",torch.zeros(2,3,4))
 with pytest.raises(ShapeRefusal):canonical_public(r)

def test_v4_guarded_loader_normalizes_actual_hdf5_shape():
 raw=record((3,4));object.__setattr__(raw,"observed_wavefield",torch.zeros(2,3,4));loader=V4GuardedOnsetLoader([raw]);normalized=loader[0];assert normalized.observed_wavefield.shape==(2,1,3,4) and canonical_public(normalized).observed_wavefield.shape==(2,3,4)

def test_actual_coeff_regularizer_and_zero_init():
 public=canonical_public(record());args=deployment_args(public,torch.zeros(401,201,201),torch.ones(201,201),torch.device("cpu"));zero=model();corrected0,coeff0=forward_actual_coefficients(zero,args);assert torch.count_nonzero(coeff0)==0
 losses0,_,_=actual_coefficient_loss(zero,args,corrected0[:,3:].detach());assert losses0["normalized_coefficient_energy"]==0
 nonzero=model();nonzero.pointwise_out.bias.data.fill_(.2);corrected,coeff=forward_actual_coefficients(nonzero,args);losses,_,_=actual_coefficient_loss(nonzero,args,corrected[:,3:].detach());assert losses["normalized_coefficient_energy"]>0
 losses["total"].backward();assert nonzero.pointwise_out.bias.grad is not None and torch.linalg.vector_norm(nonzero.pointwise_out.bias.grad)>0

def test_actual_split_ledger_and_serialized_before_truth(tmp_path):
 candidate=torch.arange(12,dtype=torch.float32).reshape(3,2,2);ledger=AccessLedger("validation");spool=CandidateSpool(tmp_path/"run");seal=spool.seal("sample",candidate,ledger)
 path=tmp_path/"truth.h5"
 with h5py.File(path,"w") as h:h.create_dataset("wavefield",data=np.zeros((1,6,2,2),np.float32))
 line,payload=auth();token=tmp_path/"run/token.json";truth=open_truth_after_spool(spool=spool,seal=seal,path=path,index=0,k1=2,mode="validation-once",split=ledger.split,ledger=ledger,authorization=payload,lineage=line,once_token=token)
 assert truth.shape[0]==3 and token.exists();events=[e["event"] for e in ledger.events];assert events.index("candidate_sealed")<events.index("future_truth_read") and next(e for e in ledger.events if e["event"]=="candidate_sealed")["serialized"] is True

def test_score_uses_same_sealed_candidate_truth_mutation_invariant(tmp_path):
 ledger=AccessLedger("train");spool=CandidateSpool(tmp_path/"run");candidate=torch.randn(4,3,3);seal=spool.seal("s",candidate,ledger);truth=torch.randn(4,3,3);before=score_same_sealed_candidate(spool,seal,ledger);truth.add_(999);after=score_same_sealed_candidate(spool,seal,ledger);assert torch.equal(before,after) and torch.equal(after,candidate)

def test_safe_cleanup_owned_only_and_refuses_escape(tmp_path):
 ledger=AccessLedger("train");spool=CandidateSpool(tmp_path/"run");seal=spool.seal("s",torch.ones(2),ledger);spool.cleanup(seal,ledger);assert not Path(seal["path"]).exists() and ledger.events[-1]["event"]=="candidate_spool_cleanup"
 outsider=tmp_path/"other.pt";torch.save({"x":1},outsider);bad={**seal,"path":str(outsider),"file_sha256":"x"}
 with pytest.raises(SpoolRefusal):spool.cleanup(bad,ledger)
 assert outsider.exists()

def test_future_refuses_missing_or_tampered_spool(tmp_path):
 ledger=AccessLedger("train");spool=CandidateSpool(tmp_path/"run");seal=spool.seal("s",torch.ones(2),ledger);Path(seal["path"]).write_bytes(b"tampered")
 with pytest.raises(SpoolRefusal):spool.validate(seal,ledger)

def production_backend(tmp_path,split="train",mode="smoke",nonzero=False):
 r=record();line=Lineage("r16_dscp_v4",mode,"r","c","cfg","p","b","bt","par","in");payload={"schema":AUTH_SCHEMA,"status":"authorized",**line.__dict__,"gates_sha256":"g"};payload["authorization_digest"]=canonical_sha(payload)
 h5=tmp_path/f"{split}.h5"
 with h5py.File(h5,"w") as h:h.create_dataset("wavefield",shape=(1,401,201,201),dtype="f4",chunks=(1,1,201,201),fillvalue=1.)
 candidate=model()
 if nonzero:candidate.pointwise_out.bias.data.fill_(.2)
 optimizer=torch.optim.AdamW(candidate.parameters(),lr=1e-3)
 backend=V4ProductionBackend(public_loader=[r],parent_predictor=lambda public:torch.zeros(401,201,201),travel_builder=lambda velocity,source:np.zeros_like(velocity),candidate=candidate,optimizer=optimizer,authorization=payload,lineage=line,source_h5=h5,run_dir=tmp_path/f"run-{split}-{mode}",device=torch.device("cpu"),split=split,family_by_sample={"sample":"uniform"},once_token=tmp_path/f"run-{split}-{mode}/once.json" if split!="train" else None)
 return backend

def test_actual_shape_production_smoke_update_and_scale_coeff(tmp_path):
 backend=production_backend(tmp_path,"train","smoke");dispatcher=V4StageDispatcher(backend);result=dispatcher.dispatch("smoke",0);assert result["loss"]>0 and result["ledger_digest"] and backend.active==[]
 scale=production_backend(tmp_path,"train","scale-probe-1",nonzero=True);before={k:v.detach().clone() for k,v in scale.candidate.state_dict().items()};measured=V4StageDispatcher(scale).dispatch("scale-probe-1",0);assert measured["coefficient_energy"]>0 and all(torch.equal(before[k],v) for k,v in scale.candidate.state_dict().items())

def test_pilot_confirm_and_validation_same_serialized_candidate(tmp_path):
 pilot=production_backend(tmp_path,"train","pilot");prepared=pilot.prepare(0,"eval");truth=pilot.open_truth(prepared);sealed_hash=prepared.seal["file_sha256"];row=pilot.score(prepared,truth);assert row["candidate_file_sha256"]==sealed_hash and row["training_only"] is True and prepared.ledger.events.index(next(e for e in prepared.ledger.events if e["event"]=="candidate_sealed"))<prepared.ledger.events.index(next(e for e in prepared.ledger.events if e["event"]=="future_truth_read"))
 validation=production_backend(tmp_path,"validation","validation-once");vrow=V4StageDispatcher(validation).dispatch("validation-once",0,phase="eval");assert vrow["training_only"] is False and validation.once_token.exists()

def test_v4_dispatcher_refuses_v3_or_foreign_backend():
 with pytest.raises(TypeError):V4StageDispatcher(object())

def test_actual_r16_global_batch_one_vs_four_update_equivalence():
 base=model();initial={k:v.detach().clone() for k,v in base.state_dict().items()};inputs=[torch.randn(1,29,5,7,generator=torch.Generator().manual_seed(100+i)) for i in range(4)];targets=[torch.randn(1,16,5,7,generator=torch.Generator().manual_seed(200+i)) for i in range(4)]
 one=model();one.load_state_dict(initial);opt1=torch.optim.AdamW(one.parameters(),lr=1e-3);opt1.zero_grad()
 for x,y in zip(inputs,targets):((one.coefficient_head(x)-y).square().mean()/4).backward()
 opt1.step()
 replicas=[]
 for x,y in zip(inputs,targets):
  replica=model();replica.load_state_dict(initial);replica.zero_grad();(replica.coefficient_head(x)-y).square().mean().backward();replicas.append(replica)
 four=model();four.load_state_dict(initial);opt4=torch.optim.AdamW(four.parameters(),lr=1e-3);opt4.zero_grad()
 for name,param in four.named_parameters():param.grad=torch.stack([dict(r.named_parameters())[name].grad for r in replicas]).mean(0)
 opt4.step();assert all(torch.allclose(a,b,rtol=0,atol=2e-7) for a,b in zip(one.parameters(),four.parameters()))

def _metric(sample,value=.1):return {"sample_id":sample,"family":"uniform","aggregate_rel_l2":value,"parent_rel_l2":.3,"nonworse":True}

def test_long_partition_cosine_resume_and_rank0_writes():
 train=list(range(8));seen=[];checkpoints=[];terminals=[];cal=[0,1,2,3]
 runner=V4LongRunner(world_size=1,rank=0,train_records=train,calibration_records=cal,zero_grad=lambda:None,backward_record=lambda r,s:seen.append((r,s)),optimizer_step=lambda:None,set_lr=lambda lr:None,evaluate_record=lambda r:_metric(str(r),.1),gather=lambda rows:rows,checkpoint=lambda state,best:checkpoints.append((state,best)),terminal=terminals.append)
 result=runner.run(LongResumeState(epoch=0,global_step=1,group_index=1),max_epochs=1);assert [r for r,s in seen]==[4,5,6,7] and result["global_step"]==2 and len(checkpoints)==2 and checkpoints[0][0].group_index==2 and len(terminals)==1
 assert frozen_global_groups(train)==[[0,1,2,3],[4,5,6,7]]

def test_long_world_decision_failure_and_commands(tmp_path):
 validate_world_size({"status":"passed","selected_gpus":1},1,"0")
 with pytest.raises(GateFailure):validate_world_size({"status":"passed","selected_gpus":4},1,"0")
 one=launch_long_command({"status":"passed","selected_gpus":1},tmp_path/"one.json");four=launch_long_command({"status":"passed","selected_gpus":4});assert "python" in one and "torchrun" in four and json.loads((tmp_path/"one.json").read_text())["launched"] is False

def test_long_rank_failure_terminal_and_foreign_dispatch_refusal():
 terminals=[];runner=V4LongRunner(world_size=1,rank=0,train_records=list(range(4)),calibration_records=[0],zero_grad=lambda:None,backward_record=lambda r,s:(_ for _ in ()).throw(RuntimeError("rankfail")),optimizer_step=lambda:None,set_lr=lambda lr:None,evaluate_record=lambda r:_metric("x"),gather=lambda rows:rows,checkpoint=lambda *a:None,terminal=terminals.append)
 with pytest.raises(RuntimeError):runner.run(max_epochs=1)
 assert terminals[0]["status"]=="failed"
 with pytest.raises(TypeError):V4LongDispatcher(object(),{"status":"passed","selected_gpus":1},"0")

def test_four_rank_partition_resume_state_and_rank0_only_writes():
 seen=[];checkpoints=[];terminals=[];full=[_metric(str(i),.1) for i in range(4)]
 def gather(rows):return full if rows and "family" in rows[0] else [{"stop":False}]
 runner=V4LongRunner(world_size=4,rank=1,train_records=list(range(8)),calibration_records=list(range(4)),zero_grad=lambda:None,backward_record=lambda r,s:seen.append((r,s)),optimizer_step=lambda:None,set_lr=lambda lr:None,evaluate_record=lambda r:_metric(str(r)),gather=gather,checkpoint=lambda *a:checkpoints.append(a),terminal=terminals.append)
 result=runner.run(LongResumeState(epoch=0,global_step=1,group_index=1),max_epochs=1);assert seen==[(5,1.)] and result["global_step"]==2 and checkpoints==[] and terminals==[]

def test_resume_directory_binding_and_completed_refusal(tmp_path):
 root=tmp_path/"run";root.mkdir();(root/"last.pt").write_bytes(b"x");(root/"run_identity.json").write_text(json.dumps({"authorization_sha256":"a","code":"c"}))
 payload=validate_resume_directory(root,authorization_sha256="a",effective_bindings={"code":"c"},load_last=lambda p:{"run_identity":{"authorization_sha256":"a"}});assert payload
 (root/"terminal.json").write_text(json.dumps({"status":"completed_train"}))
 with pytest.raises(GateFailure):validate_resume_directory(root,authorization_sha256="a",effective_bindings={"code":"c"},load_last=lambda p:{})

def test_dual_workspace_tmpfs_gates_and_four_fallback(tmp_path):
 gate1=dual_space_gate(workspace=tmp_path,tmpfs="/dev/shm",checkpoint_bytes=100,spool_max_bytes=1000,world_size=1);gate4=dual_space_gate(workspace=tmp_path,tmpfs="/dev/shm",checkpoint_bytes=100,spool_max_bytes=1000,world_size=4);assert gate4["tmpfs_required_bytes"]==4*(1000+64*1024**2) and gate1["workspace_required_bytes"]==2*1024**3+300+64*1024**2
 assert tmpfs_constrained_world({"selected_gpus":4},{"tmpfs_passed":True},{"tmpfs_passed":False})==1
 with pytest.raises(GateFailure):tmpfs_constrained_world({"selected_gpus":1},{"tmpfs_passed":False},{"tmpfs_passed":False})

def test_v4_production_source_has_no_v3_backend_or_old_loop():
 import inspect
 source=inspect.getsource(v4cli);assert "ProductionSmokePilotBackend" not in source and "V4ProductionBackend" in source and "complete_unpromoted" not in source
 config=yaml.safe_load((Path(v4cli.ROOT)/"configs/r16_dscp_v4.yaml").read_text())
 for name,command in config["commands"].items():
  if name not in {"prep","authorize_stage","scale_decide","launch_long_command"}:assert command.startswith("env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=")
