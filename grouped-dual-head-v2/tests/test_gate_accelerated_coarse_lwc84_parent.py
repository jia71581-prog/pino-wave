from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch
import scripts.gate_accelerated_coarse_lwc84_parent as gate_module

from scripts.gate_accelerated_coarse_lwc84_parent import (
    ABTruthGuard,
    assemble_truth_delta,
    bootstrap_canonical_output,
    CANDIDATE,
    EXPECTED_SELECTION_DIGESTS,
    DT_A,
    DT_B,
    energy_summary,
    environment_whitelist,
    evaluate_gates,
    formal_p95,
    prediction_qc,
    present_aware_streaming,
    pretruth_physics_qc,
    profile_relation_qc,
    raw_sequence,
    select_metadata,
    sanitize_for_json,
    selection_digests,
    source_qc,
    stability_qc,
    truth_after_pretruth_qc,
    write_failure_atomic,
    update_accumulator,
    validate_environment,
    validate_environment_schema,
    validate_invocation,
)
from scripts.reattest_frozen_fine_grid_r6_train import (
    atomic_write_bytes,
    dependency_closure,
    report_bytes_fixed_point,
    verify_dependency_manifest,
)
from saved_time_phase_operator_v4.streaming_metrics import ExactWavefieldMetricAccumulator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5')
BASE = ROOT / 'results/accelerated_coarse_lwc84_residual_r1_20260905'


def test_selection_exact_records_digests_and_no_wavefield_builder() -> None:
    frozen = json.loads((BASE/'parent_gate_selection.json').read_text())
    with h5py.File(SOURCE,'r',swmr=True) as handle:
        records=select_metadata(handle)
    assert records==frozen['records']
    assert selection_digests(records)==EXPECTED_SELECTION_DIGESTS==frozen['digests']
    assert [r['source_index'] for r in records]==[169,409,153,86,254,97,196,133,1216,1136,616,1080,788,1264,1496,1180,2160,2440,2380,2270,2360,2185,2750,2365]
    assert {f:sum(r['family']==f for r in records) for f in ('uniform','layered','marmousi')}=={'uniform':8,'layered':8,'marmousi':8}
    assert len({r['group_id'] for r in records})==24
    assert 'wavefield' not in inspect.getsource(select_metadata)


class FakeWavefield:
    def __init__(self): self.reads=[]
    def __getitem__(self,key): self.reads.append(key); return np.zeros((401,2,2),np.float32)


def rec(): return {'source_index':1,'sample_id':'s','group_id':'g','family':'uniform','sample_sha256':'a'*64}


@pytest.mark.parametrize(('mode','tags','transition_count'),[
    ('full',['B_hash2','A_hash1','B_hash1','A_hash3','A_hash2','B_hash3'],10),
    ('smoke',['A_graph','A_nongraph','B_graph','B_nongraph'],8),
])
def test_truth_guard_full_and_smoke_ordering(mode,tags,transition_count):
    guard=ABTruthGuard([rec()],mode=mode); wave=FakeWavefield(); handle={'wavefield':wave}
    with pytest.raises(PermissionError): guard.read_truth(handle,index=1,sample_id='s',split='train',family='uniform')
    for tag in tags: guard.register_hash(index=1,sample_id='s',family='uniform',tag=tag,digest=hashlib_sha(tag),native_qc={'native_dtype':'float32'})
    with pytest.raises(PermissionError): guard.read_truth(handle,index=1,sample_id='wrong',split='train',family='uniform')
    with pytest.raises(PermissionError): guard.read_truth(handle,index=1,sample_id='s',split='validation',family='uniform')
    truth=guard.read_truth(handle,index=1,sample_id='s',split='train',family='uniform'); assert truth.shape==(401,2,2)
    with pytest.raises(PermissionError): guard.read_truth(handle,index=1,sample_id='s',split='train',family='uniform')
    guard.mark_complete(1); summary=guard.summary()
    assert summary['truth_read_count']==summary['metrics_complete_count']==1
    assert len(summary['transitions'])==transition_count
    assert set(summary['tag_to_sha256']['1'])==set(tags)
    assert set(summary['per_tag_native_qc']['1'])==set(tags)


def hashlib_sha(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def test_sequences_raw72_and_p95_rank69() -> None:
    assert raw_sequence(0)==('A','B','A','B','A','B')
    assert raw_sequence(1)==('B','A','B','A','B','A')
    arms=[tag for pos in range(24) for tag in raw_sequence(pos)]
    assert arms.count('A')==arms.count('B')==72
    assert formal_p95(list(range(1,73)))==69


def test_profile_relations_stability_ratios_and_source_qc() -> None:
    profile=profile_relation_qc(); assert profile['passed']
    assert profile['static_profile_max_abs']<=1e-6 and profile['b_relation_max_abs']<=1e-6 and profile['a_relation_max_abs']<=1e-6
    stability=stability_qc({'cfl_2d':.2,'lwc_qmax':.1},{'cfl_2d':.4,'lwc_qmax':.4})
    assert stability['passed'] and stability['cfl_ratio']==2 and stability['qmax_ratio']==4
    assert source_qc(333.3,777.7)['passed']


def test_pretruth_physics_qc_vetoes_before_truth_callback() -> None:
    good_a={'cfl_2d':.2,'lwc_qmax':.1}; good_b={'cfl_2d':.4,'lwc_qmax':.4}
    metrics={'A_graph':good_a,'B_graph':good_b,'A_nongraph':good_a,'B_nongraph':good_b}
    comparisons={'A':{'relative_l2':0.,'max_abs':0.},'B':{'relative_l2':0.,'max_abs':0.}}
    calls=[]
    good=pretruth_physics_qc(mode='smoke',profile={'passed':True},source={'passed':True},result_metrics=metrics,graph_nongraph=comparisons)
    assert truth_after_pretruth_qc(qc=good,truth_callback=lambda:calls.append('truth') or 7)==7 and calls==['truth']
    for bad in [
        pretruth_physics_qc(mode='smoke',profile={'passed':False},source={'passed':True},result_metrics=metrics,graph_nongraph=comparisons),
        pretruth_physics_qc(mode='smoke',profile={'passed':True},source={'passed':True},result_metrics={**metrics,'B_graph':{'cfl_2d':1.1,'lwc_qmax':.4}},graph_nongraph=comparisons),
        pretruth_physics_qc(mode='smoke',profile={'passed':True},source={'passed':True},result_metrics={**metrics,'B_graph':{'cfl_2d':.3,'lwc_qmax':.4}},graph_nongraph=comparisons),
    ]:
        before=len(calls)
        with pytest.raises(RuntimeError,match='pre-truth'): truth_after_pretruth_qc(qc=bad,truth_callback=lambda:calls.append('forbidden'))
        assert len(calls)==before


def synthetic_metrics():
    stream={'spectrum_relative_l2':{'high':.02},'phase_correlation':.9999,'xcorr_peak_shift_cells':.05,'centroid_shift_cells':.25}
    truth_stream={'phase_correlation':.9955}
    return {
        'runtime':{'A_mean':1.0/.70,'A_p95':1.0/.75,'B_mean':.90,'B_p95':1.0},
        'B_vs_A':{'energy_aggregate':.01,'family_energy':{f:.015 for f in ('uniform','layered','marmousi')},'maximum_record':.025},
        'A_vs_truth':{'energy_aggregate':.05,'family_energy':{f:.05 for f in ('uniform','layered','marmousi')}},
        'B_vs_truth':{'energy_aggregate':.0525,'record_mean':.10,'family_record_mean':{f:.12 for f in ('uniform','layered','marmousi')},'maximum_record':.25,'family_energy':{f:.055 for f in ('uniform','layered','marmousi')}},
        'B_minus_A_truth':{'aggregate':.0025,'family':{f:.005 for f in ('uniform','layered','marmousi')},'maximum_record':.01,'count_le_0_005':20},
        'temporal_A':{b:.05 for b in ('early','middle','late')},'temporal_B':{b:.055 for b in ('early','middle','late')},
        'streaming_B_vs_A':stream,'streaming_A_truth':{'phase_correlation':.996},'streaming_B_truth':truth_stream,
    }


def test_gate_boundaries_and_energy_field_semantics() -> None:
    m=synthetic_metrics(); assert all(evaluate_gates(m,True).values())
    m=synthetic_metrics(); m['runtime']['B_mean']=.9000001; assert not evaluate_gates(m,True)['B_runtime_mean']
    m=synthetic_metrics(); m['B_minus_A_truth']['count_le_0_005']=19; assert not evaluate_gates(m,True)['truth_record_delta_count']
    rows=[{'family':'uniform','x':[1.,100.]},{'family':'layered','x':[4.,400.]},{'family':'marmousi','x':[9.,900.]}]
    summary=energy_summary(rows,'x'); assert summary['energy_aggregate']==pytest.approx((14/1400)**.5)
    assert set(summary)=={'energy_aggregate','family_count','family_energy','record_mean','family_record_mean','maximum_record','record_values'}
    one=energy_summary([{'family':'uniform','x':[1.,100.]}],'x')
    assert one['family_count']=={'uniform':1,'layered':0,'marmousi':0}
    assert one['family_energy']['layered'] is None and one['family_record_mean']['marmousi'] is None


def test_sanitize_nonfinite_and_independent_failure_writer(tmp_path:Path) -> None:
    sanitized,paths=sanitize_for_json({'x':float('nan'),'y':np.float64(np.inf),'ok':1})
    assert sanitized=={'x':None,'y':None,'ok':1} and paths==['$.x','$.y']
    output=tmp_path/'failure.json'; contaminated={'schema':'s','candidate':'c','mode':'smoke','metrics':{'bad':float('nan')},'resources':{'output_bytes':0}}
    write_failure_atomic(report=contaminated,error=RuntimeError('boom'),started=0.0,guard=None,output=output)
    payload=json.loads(output.read_text()); assert payload['status']=='invalid' and 'metrics' not in payload
    assert payload['resources']['output_bytes']==output.stat().st_size


def test_single_family_pipeline_to_strict_serializer_and_full_count_guard() -> None:
    rows=[{'family':'uniform','A':[1.,100.],'B':[4.,100.],'BA':[1.,100.]}]
    at=energy_summary(rows,'A'); bt=energy_summary(rows,'B'); ba=energy_summary(rows,'BA')
    delta=assemble_truth_delta(at,bt,require_full=False)
    assert delta['family']=={'uniform':.1,'layered':None,'marmousi':None}
    payload={'A':at,'B':bt,'BA':ba,'delta':delta,'resources':{'output_bytes':0}}
    data=report_bytes_fixed_point(payload); json.loads(data); assert b'NaN' not in data
    with pytest.raises(RuntimeError,match='before family arithmetic'):
        assemble_truth_delta(at,bt,require_full=True)


def test_full_complete_family_delta_exact() -> None:
    A={'family_count':{f:8 for f in ('uniform','layered','marmousi')},'family_energy':{'uniform':.1,'layered':.2,'marmousi':.3},'energy_aggregate':.2,'record_values':[.1]*24}
    B={'family_count':{f:8 for f in ('uniform','layered','marmousi')},'family_energy':{'uniform':.11,'layered':.22,'marmousi':.33},'energy_aggregate':.21,'record_values':[.105]*24}
    delta=assemble_truth_delta(A,B,require_full=True)
    assert delta['aggregate']==pytest.approx(.01) and delta['family']=={'uniform':pytest.approx(.01),'layered':pytest.approx(.02),'marmousi':pytest.approx(.03)}
    assert delta['count_le_0_005']==24


def test_accumulator_chunk32_and_require_unique() -> None:
    acc=ExactWavefieldMetricAccumulator(energy_floor_fraction=.01,require_unique=True,stored_time_count=401)
    row={'family':'uniform','group_id':'g','sample_id':'s'}; target=np.ones((401,2,2),np.float32); pred=target.copy()
    update_accumulator(acc,pred,target,row); result=acc.finalize()
    result=present_aware_streaming(result)
    assert result['frame_count']==401 and result['unique_time_index_count']==401 and result['record_count']==1
    assert result['family_relative_l2']['uniform']==0.0 and result['family_relative_l2']['layered'] is None and result['family_relative_l2']['marmousi'] is None
    with pytest.raises(ValueError,match='duplicate'): update_accumulator(acc,pred,target,row)


def test_prediction_qc_and_serializer(tmp_path:Path) -> None:
    value=np.zeros((401,201,201),np.float32); checked,digest,qc=prediction_qc(value); assert checked.flags.c_contiguous and len(digest)==64 and qc['native_dtype']=='float32'
    bad=value.copy(); bad[0,0,0]=1
    with pytest.raises(RuntimeError): prediction_qc(bad)
    with pytest.raises(RuntimeError,match='dtype'): prediction_qc(value.astype(np.float64))
    with pytest.raises(RuntimeError,match='C-contiguous'): prediction_qc(np.asfortranarray(value))


def test_exact_success_counts_and_zero_truth_each_once_false(tmp_path: Path) -> None:
    records=[{'source_index':i,'sample_id':f's{i}','group_id':f'g{i}','family':'uniform','sample_sha256':'a'*64} for i in range(24)]
    guard=ABTruthGuard(records,mode='full'); wave=FakeWavefield(); handle={'wavefield':wave}
    tags=['A_hash1','A_hash2','A_hash3','B_hash1','B_hash2','B_hash3']
    for row in records:
        for tag in tags: guard.register_hash(index=row['source_index'],sample_id=row['sample_id'],family='uniform',tag=tag,digest=hashlib_sha(f"{row['source_index']}:{tag}"),native_qc={'finite':True})
        guard.read_truth(handle,index=row['source_index'],sample_id=row['sample_id'],split='train',family='uniform'); guard.mark_complete(row['source_index'])
    summary=guard.summary(); assert summary['prediction_hash_count']==144 and summary['truth_read_count']==24 and len(summary['transitions'])==240 and summary['each_truth_once']
    assert not ABTruthGuard(records,mode='full').summary()['each_truth_once']
    payload={'status':'x','resources':{'output_bytes':0}}; data=report_bytes_fixed_point(payload); path=tmp_path/'r.json'; atomic_write_bytes(data,path)
    assert path.stat().st_size==payload['resources']['output_bytes']==len(data)


def test_path_status_and_smoke_prerequisite(tmp_path:Path) -> None:
    tmp_path.mkdir(exist_ok=True); paths={k:str(tmp_path/k) for k in ['preregistration','selection','source_h5','manifest','marmousi','smoke_output','full_output']}
    for k in ['selection','source_h5','manifest','marmousi']: Path(paths[k]).write_text('x')
    prereg={'schema':'accelerated_coarse_lwc84_parent_gate_preregistration_v1','candidate':CANDIDATE,'status':'static_passed_smoke_pending_audit','paths':paths,'prerequisites':{'smoke':{'path':paths['smoke_output'],'status':'pending','sha256':None}}}
    Path(paths['preregistration']).write_text(json.dumps(prereg)); args=argparse.Namespace(smoke=True,full=False,preregistration=Path(paths['preregistration']),selection=Path(paths['selection']),source_h5=Path(paths['source_h5']),manifest=Path(paths['manifest']),marmousi=Path(paths['marmousi']),output=Path(paths['smoke_output']))
    assert validate_invocation(args,prereg)[0]=='smoke'; args.output=tmp_path/'bad'
    with pytest.raises(RuntimeError): validate_invocation(args,prereg)


def _preflight_prereg(tmp_path: Path) -> tuple[Path, Path, dict]:
    prereg_path=tmp_path/'prereg.json'; output=tmp_path/'fixed.json'
    prereg={
        'schema':'accelerated_coarse_lwc84_parent_gate_preregistration_v1',
        'candidate':CANDIDATE,
        'status':'static_passed_smoke_pending_audit',
        'paths':{
            'preregistration':str(prereg_path),'smoke_output':str(output),'full_output':str(tmp_path/'full.json'),
            'selection':str(tmp_path/'selection.json'),'source_h5':str(tmp_path/'source.h5'),'manifest':str(tmp_path/'manifest.jsonl'),'marmousi':str(tmp_path/'marmousi.npy'),
        },
        'environment':{'CUBLAS_WORKSPACE_CONFIG':None,'CUDA_VISIBLE_DEVICES':'0','python':'x','torch':'x','torch_cuda':'x','cudnn':1,'numpy':'x','h5py':'x','scipy':'x','nvidia_driver':'x','gpu_model':'x'},
        'algorithm_flags':{'deterministic_algorithms':False,'cudnn_benchmark':False,'cudnn_deterministic':False},
        'prerequisites':{'smoke':{'status':'pending','path':str(output),'sha256':None}},
    }
    prereg_path.write_text(json.dumps(prereg)); return prereg_path,output,prereg


def _argv(prereg: Path, output: Path, selection: str='selection.json') -> list[str]:
    return ['gate','--smoke','--preregistration',str(prereg),'--selection',str(prereg.parent/selection),'--source-h5',str(prereg.parent/'source.h5'),'--manifest',str(prereg.parent/'manifest.jsonl'),'--marmousi',str(prereg.parent/'marmousi.npy'),'--output',str(output)]


def test_environment_failures_are_atomic_before_hdf_cuda(monkeypatch:pytest.MonkeyPatch,tmp_path:Path) -> None:
    for label,query in [('wrong',lambda:{'CUBLAS_WORKSPACE_CONFIG':'bad'}),('query',lambda:(_ for _ in ()).throw(RuntimeError('query failed')) )]:
        case=tmp_path/label; case.mkdir(); prereg,output,_=_preflight_prereg(case)
        monkeypatch.setattr(sys,'argv',_argv(prereg,output)); monkeypatch.setattr(gate_module,'environment_whitelist',query)
        monkeypatch.setattr(gate_module.h5py,'File',lambda *a,**k:(_ for _ in ()).throw(AssertionError('HDF called')))
        monkeypatch.setattr(torch.cuda,'set_device',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CUDA called')))
        with pytest.raises(RuntimeError): gate_module.main()
        payload=json.loads(output.read_text()); assert payload['status']=='invalid' and payload['partial_truth_ledger'] is None
        assert payload['validation_truth_reopened_this_stage'] is False and payload['test_id_truth_reopened_this_stage'] is False


def test_real_prereg_environment_schema_and_missing_flags_fixed_failure(monkeypatch:pytest.MonkeyPatch,tmp_path:Path) -> None:
    real=json.loads((BASE/'parent_gate_preregistration.json').read_text()); validate_environment_schema(real)
    case=tmp_path/'missing'; case.mkdir(); prereg,output,p=_preflight_prereg(case); del p['algorithm_flags']; prereg.write_text(json.dumps(p))
    monkeypatch.setattr(sys,'argv',_argv(prereg,output))
    monkeypatch.setattr(gate_module.h5py,'File',lambda *a,**k:(_ for _ in ()).throw(AssertionError('HDF called')))
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CUDA called')))
    with pytest.raises(RuntimeError,match='algorithm_flags'): gate_module.main()
    payload=json.loads(output.read_text()); assert payload['status']=='invalid'; assert payload['observed_environment'] is None and payload['expected_algorithm_flags'] is None


def test_wrong_nonoutput_path_writes_fixed_failure_wrong_output_writes_nothing(monkeypatch:pytest.MonkeyPatch,tmp_path:Path) -> None:
    case=tmp_path/'nonoutput'; case.mkdir(); prereg,output,p=_preflight_prereg(case)
    valid={**p['environment'],**p['algorithm_flags']}; monkeypatch.setattr(gate_module,'environment_whitelist',lambda:valid)
    monkeypatch.setattr(sys,'argv',_argv(prereg,output,selection='wrong.json'))
    with pytest.raises(RuntimeError,match='path override'): gate_module.main()
    assert json.loads(output.read_text())['status']=='invalid'
    case=tmp_path/'output'; case.mkdir(); prereg,canonical,_=_preflight_prereg(case); override=case/'override.json'
    monkeypatch.setattr(sys,'argv',_argv(prereg,override))
    with pytest.raises(RuntimeError,match='canonical output'): gate_module.main()
    assert not canonical.exists() and not override.exists()


def test_dependency_manifest_and_fresh_import_coverage() -> None:
    dep=json.loads((BASE/'parent_gate_dependency_manifest.json').read_text()); closure=verify_dependency_manifest(dep)
    code="import json,pathlib,sys;import scripts.gate_accelerated_coarse_lwc84_parent;root=pathlib.Path.cwd().resolve();print(json.dumps(sorted({pathlib.Path(m.__file__).resolve().relative_to(root).as_posix() for m in sys.modules.values() if getattr(m,'__file__',None) and (str(pathlib.Path(m.__file__).resolve()).startswith(str(root/'scripts')) or str(pathlib.Path(m.__file__).resolve()).startswith(str(root/'src/fno_acoustic')) or str(pathlib.Path(m.__file__).resolve()).startswith(str(root/'saved_time_phase_operator_v4')))})))"
    imported=set(json.loads(subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,capture_output=True,text=True).stdout)); covered=set(closure)|{row['path'] for row in dep['explicit_bindings'] if not Path(row['path']).is_absolute()}
    assert imported<=covered


def test_dependency_paths_placeholder() -> None:
    actual=[path.relative_to(ROOT).as_posix() for path in dependency_closure([ROOT/'scripts/gate_accelerated_coarse_lwc84_parent.py'])]
    assert actual == [
        "scripts/__init__.py",
        "scripts/audit_target5.py",
        "scripts/gate_accelerated_coarse_lwc84_parent.py",
        "scripts/gate_lwc84_cuda_graph_fine_grid_trainonly.py",
        "scripts/reattest_frozen_fine_grid_r6_train.py",
        "src/fno_acoustic/__init__.py",
        "src/fno_acoustic/ais_model_components.py",
        "src/fno_acoustic/data.py",
        "src/fno_acoustic/data_generation/__init__.py",
        "src/fno_acoustic/data_generation/cpml.py",
        "src/fno_acoustic/data_generation/free_surface.py",
        "src/fno_acoustic/data_generation/fused_lwc84.py",
        "src/fno_acoustic/data_generation/grid.py",
        "src/fno_acoustic/data_generation/lwc84.py",
        "src/fno_acoustic/data_generation/model_marmousi.py",
        "src/fno_acoustic/data_generation/restriction.py",
        "src/fno_acoustic/data_generation/ricker.py",
        "src/fno_acoustic/data_generation/solver_lwc84.py",
        "src/fno_acoustic/data_generation/solver_lwc84_fused.py",
        "src/fno_acoustic/data_generation/source.py",
        "src/fno_acoustic/data_generation/stencils.py",
        "src/fno_acoustic/data_generation/velocity_models_lwc84.py",
        "src/fno_acoustic/model.py",
        "src/fno_acoustic/model_ais_mqfno.py",
        "src/fno_acoustic/model_factorized.py",
        "src/fno_acoustic/normalization.py",
        "src/fno_acoustic/numerics/__init__.py",
        "src/fno_acoustic/numerics/drp_coefficients.py",
        "src/fno_acoustic/schema.py",
        "src/fno_acoustic/temporal_operator.py",
    ]


def test_wavefield_single_guard_surface_no_artifact_and_claim_words() -> None:
    path=ROOT/'scripts/gate_accelerated_coarse_lwc84_parent.py'; tree=ast.parse(path.read_text()); count=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and node.slice.value=='wavefield': count+=1
    assert count==1 and 'handle["wavefield"]' in inspect.getsource(ABTruthGuard.read_truth)
    source=path.read_text().lower(); assert 'torch.save' not in source and '.pt' not in source and 'prediction_output' not in source
    master=(BASE/'master_protocol.json').read_text().lower(); prereg=(BASE/'parent_gate_preregistration.json').read_text().lower()
    for forbidden in ('cfl matched','teacher','restriction equivalent'):
        assert forbidden not in master+prereg
