#!/usr/bin/env python3
"""Factorial train-only pilot for the additive pyramid-MoE U-FNO operator."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.train_transfer_dg_phase_scatter64_full_ddp import (  # noqa: E402
    FullCollection,
    atomic_checkpoint,
    atomic_json,
    sha256,
)
from scripts.train_transfer_dg_wfp_e1 import CacheCollection  # noqa: E402
from scripts.train_transfer_dg_wfp_e1d import TravelCollection, apply_phase  # noqa: E402
from saved_time_phase_operator_v4.coupled_mhc_wave import RestoredUFNOPath  # noqa: E402
from saved_time_phase_operator_v4.coupled_pyramid_moe_wave import (  # noqa: E402
    PersistentComplexMediumPyramid,
    PyramidMoECoupledWaveOperator,
    RoutedPyramidDecoder,
    SoftTop2PyramidExperts,
    parameter_count,
)
from saved_time_phase_operator_v4.muon import HybridMuonAdamW, Muon  # noqa: E402
from saved_time_phase_operator_v4.phase_carrier import (  # noqa: E402
    rotate_complex_pairs,
    travel_phase_carrier,
)
from saved_time_phase_operator_v4.wfp import BackgroundFrequencyOperator  # noqa: E402


FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
BLOCK = 8


def optimizer_for(model: torch.nn.Module, name: str, config: dict):
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=float(config["adamw_lr"]),
            weight_decay=float(config["weight_decay"]),
        )
    muon_params, adam_decay, adam_no_decay = [], [], []
    for parameter_name, parameter in model.named_parameters():
        excluded = (
            parameter_name.startswith("medium_stem.0.")
            or parameter_name.startswith("source_stem.0.")
            or parameter_name.startswith("medium_pyramid.lift.")
            or parameter_name.startswith("pyramid_decoder.source_projection.")
            or parameter_name.startswith("pyramid_decoder.scalar_projection.0.")
            or parameter_name.startswith("pyramid_decoder.scalar_projection.2.")
            or parameter_name.startswith("pyramid_decoder.experts.router.3.")
            or parameter_name.startswith("pyramid_decoder.output_projection.")
            or parameter_name.startswith("pyramid_context.2.")
            or parameter_name.startswith("trunk_basis.0.")
            or parameter_name.startswith("branch_context.2.")
            or parameter_name.startswith("medium_mionet_branch.2.")
            or parameter_name.startswith("source_mionet_branch.2.")
            or ".restored_ufno.input_projection." in parameter_name
            or ".restored_ufno.output_projection." in parameter_name
            or parameter_name.startswith("physical_head.")
            or parameter_name.startswith("cpml_head.")
            or "logits" in parameter_name
            or parameter.ndim < 2
        )
        matrix_ok = parameter.ndim >= 2 and min(
            parameter.shape[0], parameter.reshape(parameter.shape[0], -1).shape[1]
        ) > 1
        if not excluded and matrix_ok:
            muon_params.append(parameter)
        elif parameter.ndim >= 2:
            adam_decay.append(parameter)
        else:
            adam_no_decay.append(parameter)
    muon = Muon(
        [{"params": muon_params, "lr": float(config["muon_lr"]),
          "initial_lr": float(config["muon_lr"]),
          "weight_decay": float(config["weight_decay"])}],
        lr=float(config["muon_lr"]), momentum=0.95,
        weight_decay=float(config["weight_decay"]), nesterov=True, ns_steps=5,
    )
    groups = []
    if adam_decay:
        groups.append({"params": adam_decay, "lr": float(config["adamw_lr"]),
                       "initial_lr": float(config["adamw_lr"]),
                       "weight_decay": float(config["weight_decay"])})
    if adam_no_decay:
        groups.append({"params": adam_no_decay, "lr": float(config["adamw_lr"]),
                       "initial_lr": float(config["adamw_lr"]), "weight_decay": 0.0})
    adamw = torch.optim.AdamW(groups, weight_decay=0.0)
    return HybridMuonAdamW(muon, adamw)


def cosine_lr(optimizer, fraction: float, eta_ratio: float = 0.01) -> None:
    factor = eta_ratio + (1.0 - eta_ratio) * 0.5 * (1.0 + math.cos(math.pi * fraction))
    for group in optimizer.param_groups:
        initial = float(group.setdefault("initial_lr", group["lr"]))
        group["lr"] = initial * factor


class PilotData:
    def __init__(self, residual: FullCollection, base: CacheCollection,
                 travel: TravelCollection, manifest: dict) -> None:
        self.residual = residual
        self.base = base
        self.travel = travel
        self.residual_pos = {row[2]: i for i, row in enumerate(residual.records)}
        self.base_pos = {row[2]: i for i, row in enumerate(base.records)}
        self.roles = defaultdict(list)
        for row in manifest["records"]:
            self.roles[row["role"]].append(self.residual_pos[row["sample_id"]])

    def block(self, position: int, start: int, device: torch.device) -> dict:
        frequencies = range(start, start + BLOCK)
        items = [self.residual.build(position, frequency, device) for frequency in frequencies]
        sample_id = self.residual.records[position][2]
        base_position = self.base_pos[sample_id]
        travel_physical, travel_exterior, _, auxiliary_total = self.travel.read(sample_id)
        auxiliary = []
        for frequency in frequencies:
            auxiliary.append(self.base.targets(base_position, frequency)[1])
        static = self.residual.static(position)
        file_index, local, _, family = self.residual.records[position]
        handle = self.residual.handles[file_index]
        return {
            "sample_id": sample_id,
            "family": family,
            "medium": torch.cat([item["medium"] for item in items])[None],
            "source": torch.cat([item["source"] for item in items])[None],
            "scalars": torch.cat([item["scalars"] for item in items])[None],
            "travel_physical": torch.from_numpy(travel_physical)[None].expand(BLOCK, -1, -1).to(device),
            "travel_exterior": torch.from_numpy(travel_exterior)[None].expand(BLOCK, -1, -1).to(device),
            "frequency_hz": torch.cat([item["frequency_hz"] for item in items]),
            "residual": torch.cat([item["residual"] for item in items]),
            "auxiliary_target": torch.from_numpy(np.stack(auxiliary)).to(device),
            "target_total": float(static["target_total"]),
            "auxiliary_total": float(auxiliary_total),
            "full_total": float(handle["full_target_time_square_norm"][local]),
            "unmodeled": float(handle["unmodeled_time_square_norm"][local]),
        }


@torch.inference_mode()
def absolute_target(parent, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    medium = batch["medium"][0, :, :12]
    source = batch["source"][0]
    scalars = batch["scalars"][0]
    physical, auxiliary = parent(medium, source, scalars)
    physical, auxiliary = apply_phase(
        physical, auxiliary, batch["travel_physical"], batch["travel_exterior"],
        batch["frequency_hz"], True,
    )
    return physical + batch["residual"], auxiliary


def model_prediction(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = batch["frequency_hz"].shape[0]
    device = batch["medium"].device
    x = (torch.arange(241, device=device, dtype=torch.float32) * 10.0 - 200.0) / 2000.0
    z = (torch.arange(221, device=device, dtype=torch.float32) * 10.0) / 2000.0
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    coordinates = torch.stack((xx, zz), dim=0)[None].expand(frequencies, -1, -1, -1)
    frequency = (batch["frequency_hz"] / 200.0)[:, None, None, None].expand(-1, 1, 221, 241)
    travel = batch["travel_exterior"][:, None]
    carrier = travel_phase_carrier(batch["travel_exterior"], batch["frequency_hz"])
    cpml_profiles = batch["medium"][0, :, 3:11]
    trunk = torch.cat((coordinates, frequency, travel, carrier, cpml_profiles), dim=1)[None]
    physical, auxiliary = model(
        batch["medium"], batch["source"], batch["scalars"], trunk
    )
    physical = physical[0]
    auxiliary = auxiliary[0]
    physical = rotate_complex_pairs(
        physical,
        travel_phase_carrier(batch["travel_physical"], batch["frequency_hz"]),
    )
    auxiliary = rotate_complex_pairs(
        auxiliary,
        travel_phase_carrier(batch["travel_exterior"], batch["frequency_hz"]),
    )
    return physical, auxiliary


def loss_terms(prediction, target, auxiliary, batch, config: dict) -> dict:
    sample_error = (prediction.float() - target.float()).square().sum((1, 2, 3))
    energy = 8.0 * sample_error.sum() / max(batch["target_total"], 1e-8)
    target_frequency = target.float().square().sum((1, 2, 3))
    floor = float(config["frequency_floor_fraction"]) * batch["target_total"] / 64.0
    balanced = (sample_error / target_frequency.add(floor).clamp_min(1e-8)).mean()
    pred_delta = prediction[1:].float() - prediction[:-1].float()
    target_delta = target[1:].float() - target[:-1].float()
    derivative_error = (pred_delta - target_delta).square().sum((1, 2, 3))
    derivative_target = target_delta.square().sum((1, 2, 3))
    continuity = (derivative_error / derivative_target.add(floor).clamp_min(1e-8)).mean()
    auxiliary_error = (auxiliary.float() - batch["auxiliary_target"].float()).square().sum()
    cpml = 8.0 * auxiliary_error / max(batch["auxiliary_total"], 1e-8)
    total = (energy + float(config["balanced_weight"]) * balanced
             + float(config["continuity_weight"]) * continuity
             + float(config["cpml_weight"]) * cpml)
    return {"total": total, "energy": energy, "balanced": balanced,
            "continuity": continuity, "cpml": cpml}


@torch.inference_mode()
def evaluate(model, parent, data: PilotData, positions: list[int], device) -> dict:
    model.eval(); rows=[]
    for position in positions:
        candidate_low = parent_low = 0.0
        full_total = unmodeled = None
        family = sample_id = None
        for start in range(0, 64, BLOCK):
            batch = data.block(position, start, device)
            target, _ = absolute_target(parent, batch)
            prediction, _ = model_prediction(model, batch)
            weights = torch.full((BLOCK, 1, 1, 1), 2.0, device=device, dtype=torch.float64)
            if start == 0: weights[0] = 1.0
            candidate_low += float(((prediction.double()-target.double()).square()*weights).sum())
            parent_prediction = target - batch["residual"]
            parent_low += float(((parent_prediction.double()-target.double()).square()*weights).sum())
            full_total=batch["full_total"];unmodeled=batch["unmodeled"]
            family=batch["family"];sample_id=batch["sample_id"]
        candidate=math.sqrt((candidate_low+unmodeled)/max(full_total,1e-300))
        baseline=math.sqrt((parent_low+unmodeled)/max(full_total,1e-300))
        rows.append({"sample_id":sample_id,"family":family,
                     "candidate":candidate,"parent":baseline,
                     "improvement":1-candidate/max(baseline,1e-300)})
    return {
        "record_count":len(rows),
        "candidate_mean":float(np.mean([r["candidate"] for r in rows])),
        "parent_mean":float(np.mean([r["parent"] for r in rows])),
        "per_family_candidate":{f:float(np.mean([r["candidate"] for r in rows if r["family"]==f])) for f in FAMILIES},
        "per_family_parent":{f:float(np.mean([r["parent"] for r in rows if r["family"]==f])) for f in FAMILIES},
        "nonworse_count":sum(r["candidate"]<=r["parent"] for r in rows),
        "rows":rows,
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-cache",type=Path,action="append",required=True)
    parser.add_argument("--base-cache",type=Path,action="append",required=True)
    parser.add_argument("--travel",type=Path,action="append",required=True)
    parser.add_argument("--full-manifest",type=Path,required=True)
    parser.add_argument("--pilot-manifest",type=Path,required=True)
    parser.add_argument("--preregistration",type=Path,required=True)
    parser.add_argument("--parent-checkpoint",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--use-mhc",action="store_true")
    parser.add_argument("--optimizer",choices=("adamw","muon"),required=True)
    parser.add_argument("--seed",type=int,default=372)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    prereg=json.loads(args.preregistration.read_text());config=prereg["training"]
    bindings=prereg["bindings"]
    if sha256(Path(__file__))!=bindings["trainer_sha256"]:
        raise RuntimeError("factorial trainer binding drift")
    if sha256(ROOT/"saved_time_phase_operator_v4/coupled_pyramid_moe_wave.py")!=bindings["model_sha256"]:
        raise RuntimeError("pyramid-MoE model binding drift")
    if sha256(ROOT/"saved_time_phase_operator_v4/coupled_mhc_wave.py")!=bindings["base_model_sha256"]:
        raise RuntimeError("inherited coupled base-model binding drift")
    if sha256(args.pilot_manifest)!=bindings["pilot_manifest_sha256"]:
        raise RuntimeError("fresh pilot manifest binding drift")
    if sha256(args.parent_checkpoint)!=bindings["parent_checkpoint_sha256"]:
        raise RuntimeError("parent checkpoint binding drift")
    arm=("mhc" if args.use_mhc else "plain")+"_"+args.optimizer
    if arm not in prereg["arms"]:raise RuntimeError("unregistered factorial arm")
    full_manifest=json.loads(args.full_manifest.read_text())
    pilot_manifest=json.loads(args.pilot_manifest.read_text())
    residual=FullCollection(args.residual_cache,full_manifest)
    base=CacheCollection(args.base_cache,full_manifest,expected_count=2800)
    travel=TravelCollection(args.travel,expected_count=2800)
    data=PilotData(residual,base,travel,pilot_manifest)
    device=torch.device("cuda");torch.manual_seed(args.seed);np.random.seed(args.seed)
    parent=BackgroundFrequencyOperator(medium_channels=12,source_channels=5,width=64,rank=32,depth=6,arm="wfp",radii=(1,2,3,4,5,6)).to(device)
    parent.load_state_dict(torch.load(args.parent_checkpoint,map_location="cpu",weights_only=False)["model_state"])
    parent.eval();parent.requires_grad_(False)
    model=PyramidMoECoupledWaveOperator(use_mhc=args.use_mhc).to(device)
    if not 25_000_000 <= parameter_count(model) <= 27_000_000:raise RuntimeError("restored model is outside 26M budget")
    if sum(isinstance(module,RestoredUFNOPath) for module in model.modules())!=4:
        raise RuntimeError("restored model must contain four complete U-FNO paths")
    if sum(isinstance(module,PersistentComplexMediumPyramid) for module in model.modules())!=1:
        raise RuntimeError("restored model must contain one persistent four-level pyramid")
    if sum(isinstance(module,RoutedPyramidDecoder) for module in model.modules())!=1:
        raise RuntimeError("restored model must contain one all-level pyramid decoder")
    if sum(isinstance(module,SoftTop2PyramidExperts) for module in model.modules())!=1:
        raise RuntimeError("restored model must contain one shared/routed expert bank")
    optimizer=optimizer_for(model,args.optimizer,config)
    fit=data.roles["fit"];calibration=data.roles["calibration"];confirmation=data.roles["confirmation"]
    by_family=defaultdict(list)
    for p in fit:by_family[residual.records[p][3]].append(p)
    steps_per_epoch=max(len(by_family[f])*8 for f in FAMILIES)
    total_updates=int(config["epochs"])*steps_per_epoch
    identity={"schema":"transfer_dg_coupled_mhc_muon_pilot_lane_v1","arm":arm,
              "use_mhc":args.use_mhc,"optimizer":args.optimizer,"seed":args.seed,
              "parameter_count":parameter_count(model),"epochs":config["epochs"],
              "steps_per_epoch":steps_per_epoch,"total_updates":total_updates,
              "frequency_block":BLOCK,"persistent_pyramid_levels":4,
              "shared_experts":2,"routed_experts":4,"expert_top_k":2,
              "router_weight":config["router_weight"],
              "pilot_manifest_sha256":sha256(args.pilot_manifest),
              "trainer_sha256":sha256(Path(__file__)),"validation_opened":False,"test_id_opened":False}
    atomic_json(identity,args.output_dir/"run_identity.json")
    rng=np.random.default_rng(args.seed+701);best=None;update=0;started=time.time()
    metrics=args.output_dir/"metrics.jsonl"
    for epoch in range(1,int(config["epochs"])+1):
        schedules={}
        for family in FAMILIES:
            values=np.asarray([(p,b) for p in by_family[family] for b in range(0,64,BLOCK)],dtype=np.int64)
            rng.shuffle(values);schedules[family]=values
        model.train()
        for step in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True);sums=defaultdict(float)
            for family in FAMILIES:
                position,start=schedules[family][step%len(schedules[family])]
                batch=data.block(int(position),int(start),device)
                target,_=absolute_target(parent,batch)
                prediction,auxiliary=model_prediction(model,batch)
                terms=loss_terms(prediction,target,auxiliary,batch,config)
                terms["router"] = model.router_auxiliary_loss()
                terms["total"] = terms["total"] + float(config["router_weight"]) * terms["router"]
                (terms["total"]/len(FAMILIES)).backward()
                for key,value in terms.items():sums[key]+=float(value.detach())/len(FAMILIES)
                for key,value in model.route_statistics().items():
                    sums[f"{family}_{key}"] = float(value)
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["gradient_clip_norm"]))
            optimizer.step();update+=1;cosine_lr(optimizer,update/total_updates)
            if update%20==0:
                event={"event":"update","epoch":epoch,"update":update,**sums,"elapsed_s":time.time()-started}
                with metrics.open("a") as h:h.write(json.dumps(event,sort_keys=True)+"\n")
                print(json.dumps(event),flush=True)
        cal=evaluate(model,parent,data,calibration,device);score=cal["candidate_mean"]
        with metrics.open("a") as h:h.write(json.dumps({"event":"calibration","epoch":epoch,"update":update,"metrics":cal},sort_keys=True)+"\n")
        if best is None or score<best["score"]:
            best={"score":score,"epoch":epoch,"update":update,"metrics":cal}
            atomic_checkpoint({"model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},"identity":identity,"epoch":epoch,"update":update,"calibration":cal},args.output_dir/"best.pt")
        print(json.dumps({"event":"calibration","epoch":epoch,"candidate":score,"parent":cal["parent_mean"]}),flush=True)
    checkpoint=torch.load(args.output_dir/"best.pt",map_location="cpu",weights_only=False);model.load_state_dict(checkpoint["model_state"]);model.to(device)
    confirm=evaluate(model,parent,data,confirmation,device)
    terminal={"schema":"transfer_dg_coupled_mhc_muon_pilot_lane_terminal_v1","status":"complete","arm":arm,
              "best_epoch":best["epoch"],"best_update":best["update"],"calibration":best["metrics"],
              "confirmation":confirm,"checkpoint":str((args.output_dir/"best.pt").resolve()),
              "checkpoint_sha256":sha256(args.output_dir/"best.pt"),"elapsed_s":time.time()-started,
              "validation_opened":False,"test_id_opened":False}
    atomic_json(terminal,args.output_dir/"terminal.json")
    residual.close();base.close();travel.close();return 0


if __name__=="__main__":raise SystemExit(main())
