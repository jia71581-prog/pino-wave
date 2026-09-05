#!/usr/bin/env python
"""Full-support late-ASAM refinement for the V4 dense decoder."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.data.pilot import make_pilot_loader
from grouped_ufno_mionet_v3.training.checkpoint import (
    load_checkpoint,
    save_checkpoint_atomic,
)
from saved_time_phase_operator_v4.asam import (
    asam_perturb,
    asam_rho_for_update,
    asam_validation_decision,
    build_asam_adamw,
    freeze_for_asam,
)
from saved_time_phase_operator_v4.data import (
    ExactStoredTimeBatchDataset,
    merge_pilot_batches,
)
from saved_time_phase_operator_v4.full_support import build_full_support_schedule
from saved_time_phase_operator_v4.lbfgs import refinement_gate
from scripts.refine_saved_time_v4_lbfgs import (
    _append_jsonl,
    _gpu_snapshot,
    _load_bound_model,
)
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_full_support import (
    _evaluate as _evaluate_full_support,
    _train_update as _full_support_loss_backward,
    validation_panel_indices,
)
from scripts.train_saved_time_v4_probe import (
    _atomic_hardlink,
    _atomic_json,
    _digest,
)


def _loss_backward(model, optimizer, batch, normalizer, device, model_config) -> float:
    values = _full_support_loss_backward(
        model,
        optimizer,
        batch,
        normalizer,
        device,
        model_config,
        microbatch_records=int(model_config["microbatch_records"]),
    )
    total = float(values["total"])
    if not math.isfinite(total):
        raise FloatingPointError("non-finite ASAM full-support loss")
    return total


def _training_loader(config, model_config, base, manifest, smoke_updates):
    updates = int(smoke_updates) if smoke_updates else int(config["updates"])
    macro_count = updates * int(config["macros_per_update"])
    macro_records = int(model_config["macro_records"])
    macros_per_epoch = math.ceil(
        int(base.data.expected_train_records)
        / (macro_records * int(config["macros_per_update"]))
    ) * int(config["macros_per_update"])
    schedule_epochs = math.ceil(macro_count / macros_per_epoch)
    schedule = build_full_support_schedule(
        int(base.data.expected_train_records),
        epochs=schedule_epochs,
        macro_records=macro_records,
        macros_per_update=int(config["macros_per_update"]),
        seed=int(config["train_seed"]),
    )[:macro_count]
    if not smoke_updates:
        seen = {index for spec in schedule for index in spec.record_indices}
        expected = int(base.data.expected_train_records)
        if len(seen) != expected:
            raise RuntimeError(f"ASAM schedule covers {len(seen)} of {expected} train records")
    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5, manifest, split="train", schedule=schedule,
        query_points=int(config["query_points"]), seed=int(config["train_seed"]),
        time_policy=str(model_config.get("time_policy", "appearance16")),
        frames_per_record=int(model_config.get("training_frames_per_record", 24)),
        travel_time_h5=model_config.get("travel_time_h5"),
    )
    loader = make_pilot_loader(
        dataset, workers=0 if smoke_updates else int(config["workers"]),
        prefetch_factor=int(config["prefetch_factor"]), pin_memory=True,
    )
    return updates, loader, schedule


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--smoke-updates",type=int,default=0); args=parser.parse_args(argv)
    config=yaml.safe_load(Path(args.config).read_text()); device=torch.device("cuda")
    model,base,manifest,parent_identity=_load_bound_model(config,device)
    model_config = parent_identity.get("model_config", parent_identity["config"])
    if not isinstance(model_config, dict):
        raise ValueError("ASAM parent has no full-support model config")
    parameters=freeze_for_asam(
        model,
        trainable_prefixes=tuple(config.get("trainable_prefixes", ("dense_decoder",))),
    ); normalizer=load_normalizer(base,manifest.digest)
    named_parameters=tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if {id(parameter) for _, parameter in named_parameters} != {
        id(parameter) for parameter in parameters
    }:
        raise RuntimeError("ASAM named parameter selection does not match the frozen stage")
    updates,loader,schedule=_training_loader(
        config,model_config,base,manifest,args.smoke_updates
    )
    validation_indices=validation_panel_indices(
        validation_records=int(base.data.expected_validation_records),
        panel_records=int(config["validation_records"]),
        # validation_panel_indices uses a one-based epoch contract.  Keep the
        # parent baseline and every ASAM candidate on the same first panel.
        epoch=1,
        seed=int(config["validation_seed"]),
    )
    def evaluate_current():
        return _evaluate_full_support(
            model,base,manifest,normalizer,device,model_config,validation_indices,
            epoch_offset=0,time_policy="validation_fixed",
            frames_per_record=int(config["validation_frames_per_record"]),
        )
    if bool(config.get("recompute_parent_baseline", False)):
        model.eval()
        parent_metrics=evaluate_current()
    else:
        parent_metrics=json.loads(Path(config["parent_best_report"]).read_text())["metrics"]
    train_records=tuple(record for record in manifest.records if record.split=="train")
    identity={"schema":"saved_time_v4_asam_full_support_v2","config":config,"model_config":parent_identity.get("model_config",parent_identity["config"]),"parent_run_digest":parent_identity["run_digest"],"manifest_digest":manifest.digest,"train_schedule_sample_ids":[[train_records[index].sample_id for index in spec.record_indices] for spec in schedule]}
    identity["run_digest"]=_digest(identity)
    root=Path(config["artifact_dir"])/("smoke" if args.smoke_updates else "run"); root.mkdir(parents=True,exist_ok=True)
    identity_path=root/"run_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity: raise ValueError("ASAM run identity mismatch")
    if not identity_path.exists(): _atomic_json(identity,identity_path)
    opt=config["optimizer"]
    optimizer=build_asam_adamw(
        named_parameters,
        learning_rate=float(opt["learning_rate"]),
        weight_decay=float(opt["weight_decay"]),
        betas=(float(opt.get("beta1", 0.9)), float(opt.get("beta2", 0.999))),
        eps=float(opt.get("eps", 1.0e-8)),
        learning_rates_by_prefix=opt.get("learning_rates_by_prefix"),
    )
    maximum_rho=float(opt["rho"])
    minimum_rho=float(opt.get("minimum_rho", maximum_rho))
    registered_rho_warmup_updates=int(opt.get("rho_warmup_updates", 0))
    rho_warmup_updates=min(registered_rho_warmup_updates,max(0,updates-1))
    perturb_parameter_minimum_ndim=int(
        opt.get("perturb_parameter_minimum_ndim", 0)
    )
    rejection_backoff=float(opt.get("rejection_backoff", 0.5))
    minimum_multiplier=float(opt.get("minimum_multiplier", 1.0 / 16.0))
    lr_multiplier=1.0; rho_multiplier=1.0
    accepted_score=float(parent_metrics["aggregate_relative_l2"])
    accepted_family=dict(parent_metrics["family_relative_l2"])
    baseline_gate=refinement_gate(
        parent_score=accepted_score,
        candidate_score=accepted_score,
        parent_family=accepted_family,
        candidate_family=accepted_family,
        minimum_relative_improvement=float(config["gate"]["minimum_relative_improvement"]),
        family_regression_tolerance=float(config["gate"]["family_regression_tolerance"]),
        maximum_candidate_score=config["gate"].get("maximum_candidate_score"),
        maximum_candidate_family_score=config["gate"].get(
            "maximum_candidate_family_score"
        ),
    )
    baseline_checkpoint=root/"checkpoints"/"update_000.pt"
    save_checkpoint_atomic(
        baseline_checkpoint,model=model,optimizer=optimizer,epoch=0,global_step=0,
        manifest_digest=manifest.digest,config_digest=identity["run_digest"],
        metrics={"validation":accepted_score},
    )
    _atomic_hardlink(baseline_checkpoint,root/"latest.pt")
    _atomic_hardlink(baseline_checkpoint,root/"best.pt")
    best={
        "event":"asam_baseline","update":0,"metrics":parent_metrics,
        "gate":baseline_gate,"checkpoint":str(baseline_checkpoint),
        "baseline_recomputed":bool(config.get("recompute_parent_baseline",False)),
    }
    _atomic_json(best,root/"best.json")
    iterator=iter(loader); torch.cuda.reset_peak_memory_stats()
    for update in range(1,updates+1):
        macros=tuple(next(iterator) for _ in range(int(config["macros_per_update"])))
        batch=merge_pilot_batches(macros); model.train(); optimizer.zero_grad(set_to_none=True)
        scheduled_learning_rate=asam_rho_for_update(
            update,total_updates=updates,
            maximum_rho=float(opt["learning_rate"]),
            minimum_rho=float(opt["minimum_learning_rate"]),warmup_updates=0,
        )*lr_multiplier
        learning_rate_factor=(
            scheduled_learning_rate / float(opt["learning_rate"])
        )
        for group in optimizer.param_groups:
            group["lr"]=float(group["initial_lr"])*learning_rate_factor
        first_loss=_loss_backward(model,optimizer,batch,normalizer,device,model_config)
        scheduled_rho=asam_rho_for_update(
            update,total_updates=updates,maximum_rho=maximum_rho,
            minimum_rho=minimum_rho,warmup_updates=rho_warmup_updates,
        )*rho_multiplier
        perturbation=asam_perturb(
            parameters,rho=scheduled_rho,eta=float(opt["eta"]),
            minimum_parameter_ndim=perturb_parameter_minimum_ndim,
        )
        try:
            optimizer.zero_grad(set_to_none=True)
            second_loss=_loss_backward(model,optimizer,batch,normalizer,device,model_config)
        finally:
            perturbation.restore()
        gradient_norm=float(torch.nn.utils.clip_grad_norm_(parameters,float(opt["gradient_clip"])))
        optimizer.step()
        should_evaluate=update%int(config["evaluation_every"])==0 or update==updates or bool(args.smoke_updates)
        if not should_evaluate: continue
        metrics=evaluate_current()
        score=float(metrics["aggregate_relative_l2"])
        gate=refinement_gate(parent_score=float(parent_metrics["aggregate_relative_l2"]),candidate_score=score,parent_family=parent_metrics["family_relative_l2"],candidate_family=metrics["family_relative_l2"],minimum_relative_improvement=float(config["gate"]["minimum_relative_improvement"]),family_regression_tolerance=float(config["gate"]["family_regression_tolerance"]),maximum_candidate_score=config["gate"].get("maximum_candidate_score"),maximum_candidate_family_score=config["gate"].get("maximum_candidate_family_score"))
        decision=asam_validation_decision(
            accepted_score=accepted_score,candidate_score=score,
            accepted_family=accepted_family,
            candidate_family=metrics["family_relative_l2"],
            family_regression_tolerance=float(config["gate"]["family_regression_tolerance"]),
            learning_rate_multiplier=lr_multiplier,rho_multiplier=rho_multiplier,
            rejection_backoff=rejection_backoff,minimum_multiplier=minimum_multiplier,
        )
        peak=int(torch.cuda.max_memory_allocated())
        if peak>float(config["gate"]["maximum_peak_cuda_gib"])*1024**3: raise RuntimeError("ASAM peak CUDA memory exceeded the registered limit")
        checkpoint=root/"checkpoints"/f"update_{update:03d}.pt"
        save_checkpoint_atomic(checkpoint,model=model,optimizer=optimizer,epoch=update,global_step=update,manifest_digest=manifest.digest,config_digest=identity["run_digest"],metrics={"validation":score})
        report={"event":"asam_evaluation","update":update,"first_loss":first_loss,"sharp_loss":second_loss,"asam_gradient_norm":perturbation.gradient_norm,"perturbation_norm":perturbation.perturbation_norm,"update_gradient_norm":gradient_norm,"rho":scheduled_rho,"rho_multiplier":rho_multiplier,"learning_rates":{str(group["group_name"]):float(group["lr"]) for group in optimizer.param_groups},"learning_rate_multiplier":lr_multiplier,"metrics":metrics,"gate":gate,"acceptance":{"accepted":decision.accepted,"score_improved":decision.score_improved,"family_safe":decision.family_safe,"next_learning_rate_multiplier":decision.learning_rate_multiplier,"next_rho_multiplier":decision.rho_multiplier},"peak_cuda_bytes":peak,"gpu":_gpu_snapshot(),"checkpoint":str(checkpoint)}
        _append_jsonl(root/"evaluations.jsonl",report)
        if decision.accepted:
            accepted_score=score
            accepted_family=dict(metrics["family_relative_l2"])
            _atomic_hardlink(checkpoint,root/"latest.pt")
            if score<float(best["metrics"]["aggregate_relative_l2"]):
                best=report; _atomic_hardlink(checkpoint,root/"best.pt"); _atomic_json(best,root/"best.json")
        else:
            load_checkpoint(
                root/"latest.pt",model=model,optimizer=optimizer,
                expected_manifest_digest=manifest.digest,
                expected_config_digest=identity["run_digest"],restore_rng=False,
                map_location=device,
            )
        lr_multiplier=decision.learning_rate_multiplier
        rho_multiplier=decision.rho_multiplier
        print(json.dumps(report,sort_keys=True),flush=True)
    validation_passed=bool(best["gate"]["passed"])
    terminal={"status":"complete","best":best,"gate":best["gate"],"run_digest":identity["run_digest"],"same_protocol_validation_passed":validation_passed,"claim":("ASAM target achieved on same-protocol validation" if validation_passed else "ASAM finished; absolute accuracy target not yet achieved")}; _atomic_json(terminal,root/"terminal.json"); print(json.dumps(terminal,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
