#!/usr/bin/env python
"""Resumable 2x2 saved-time capacity-by-phase probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Mapping

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.data.pilot import PilotStepSpec, build_pilot_schedule, make_pilot_loader
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint, save_checkpoint_atomic
from saved_time_phase_operator_v4.data import ExactStoredTimeBatchDataset, split_pilot_batch
from saved_time_phase_operator_v4.metrics import exact_wavefield_metrics
from saved_time_phase_operator_v4.losses import band_limited_residual_loss, frame_relative_l2
from saved_time_phase_operator_v4.operator import SavedTimePhaseOperatorV4, load_v3_backbone
from saved_time_phase_operator_v4.probe import probe_variants, select_probe_candidate
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_hardlink(source: Path, destination: Path) -> None:
    """Point a stable checkpoint name at source without a missing-file window."""
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    try:
        os.link(source, partial)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _schedule(manifest, *, split: str, pool_steps: int, total_steps: int, seed: int, offset: int = 0):
    pool = build_pilot_schedule(manifest, split=split, steps=pool_steps, seed=seed)
    return tuple(
        PilotStepSpec(step=offset + step, record_indices=pool[step % len(pool)].record_indices)
        for step in range(total_steps)
    )


def _model(base: V3Config, manifest, variant) -> SavedTimePhaseOperatorV4:
    cfg = base.model
    token_grid = round(cfg.token_count**0.5)
    return SavedTimePhaseOperatorV4(
        saved_time_s=manifest.time_s, width=cfg.width, rank=cfg.mionet_rank,
        spectral_rank=cfg.spectral_rank, modes=cfg.modes,
        dense_spectral_rank=variant.spectral_rank, dense_modes=variant.modes,
        dense_depth=variant.depth, dense_time_block=1,
        dense_coupled_axes=variant.coupled_axes,
        dense_local_differential_residual=variant.local_differential_residual,
        dense_coupled_2d_rank=variant.coupled_2d_rank,
        dense_temporal_basis_rank=variant.temporal_basis_rank,
        dense_family_expert_rank=variant.family_expert_rank,
        dense_band_adapter_rank=variant.band_adapter_rank,
        dense_band_adapter_architecture=variant.band_adapter_architecture,
        dense_band_adapter_spectral_rank=variant.band_adapter_spectral_rank,
        dense_band_adapter_modes=variant.band_adapter_modes,
        dense_band_adapter_full_depth=variant.band_adapter_full_depth,
        dense_band_adapter_coarse_depth=variant.band_adapter_coarse_depth,
        dense_band_adapter_activation_checkpointing=(
            variant.band_adapter_activation_checkpointing
        ),
        dense_band_adapter_dropout=variant.band_adapter_dropout,
        dense_band_adapter_preserve_high_band=(
            variant.band_adapter_preserve_high_band
        ),
        dense_local_field=variant.local_field,
        dense_local_field_channel_multipliers=variant.local_field_channel_multipliers,
        dense_local_field_causal_width_s=variant.local_field_causal_width_s,
        dense_local_field_residual=variant.local_field_residual,
        dense_local_field_extended_late_features=getattr(
            variant, "local_field_extended_late_features", False
        ),
        dense_local_field_temporal_operator_rank=getattr(
            variant, "local_field_temporal_operator_rank", 0
        ),
        dense_local_field_temporal_operator_spatial_kernel=getattr(
            variant, "local_field_temporal_operator_spatial_kernel", 1
        ),
        dense_local_field_warp=getattr(variant, "local_field_warp", False),
        dense_local_field_warp_max_shift_cells=getattr(
            variant, "local_field_warp_max_shift_cells", 8.0
        ),
        dense_local_field_warp_shift_dilation=getattr(
            variant, "local_field_warp_shift_dilation", 1
        ),
        dense_local_field_green_kernel=getattr(variant, "local_field_green_kernel", False),
        dense_local_field_green_kernel_size=getattr(
            variant, "local_field_green_kernel_size", 5
        ),
        dense_local_field_green_dilations=getattr(
            variant, "local_field_green_dilations", (1, 2, 4)
        ),
        dense_local_field_temporal_latent_basis=getattr(
            variant, "local_field_temporal_latent_basis", False
        ),
        dense_local_field_temporal_latent_rank=getattr(
            variant, "local_field_temporal_latent_rank", 8
        ),
        dense_local_field_temporal_latent_harmonics=getattr(
            variant, "local_field_temporal_latent_harmonics", 4
        ),
        dense_local_field_multi_arrival=getattr(
            variant, "local_field_multi_arrival", False
        ),
        dense_local_field_multi_arrival_paths=getattr(
            variant, "local_field_multi_arrival_paths", 3
        ),
        dense_local_field_multi_arrival_max_shift_cells=getattr(
            variant, "local_field_multi_arrival_max_shift_cells", 8.0
        ),
        dense_local_field_multi_arrival_max_delay_frac=getattr(
            variant, "local_field_multi_arrival_max_delay_frac", 0.5
        ),
        dense_local_field_dispersive_modal=getattr(
            variant, "local_field_dispersive_modal", False
        ),
        dense_local_field_dispersive_modal_modes=getattr(
            variant, "local_field_dispersive_modal_modes", 16
        ),
        dense_local_field_dispersive_modal_max_frequency=getattr(
            variant, "local_field_dispersive_modal_max_frequency", 8.0
        ),
        dense_local_field_windowed_propagation=getattr(
            variant, "local_field_windowed_propagation", False
        ),
        dense_local_field_windowed_propagation_window=getattr(
            variant, "local_field_windowed_propagation_window", 2
        ),
        dense_local_field_windowed_propagation_stride=getattr(
            variant, "local_field_windowed_propagation_stride", 1
        ),
        dense_local_field_windowed_propagation_rank=getattr(
            variant, "local_field_windowed_propagation_rank", 8
        ),
        dense_local_field_windowed_propagation_max_advect_cells=getattr(
            variant, "local_field_windowed_propagation_max_advect_cells", 8.0
        ),
        dense_local_field_helmholtz_synthesis=getattr(
            variant, "local_field_helmholtz_synthesis", False
        ),
        dense_local_field_helmholtz_synthesis_frequencies=getattr(
            variant, "local_field_helmholtz_synthesis_frequencies", 64
        ),
        dense_local_field_helmholtz_synthesis_wkb_phase=getattr(
            variant, "local_field_helmholtz_synthesis_wkb_phase", True
        ),
        dense_local_field_helmholtz_synthesis_rank=getattr(
            variant, "local_field_helmholtz_synthesis_rank", 0
        ),
        dense_local_field_helmholtz_synthesis_late_rank=getattr(
            variant, "local_field_helmholtz_synthesis_late_rank", 0
        ),
        dense_local_field_helmholtz_synthesis_late_frequencies=getattr(
            variant, "local_field_helmholtz_synthesis_late_frequencies", 0
        ),
        dense_local_field_helmholtz_synthesis_frequency_softmax=getattr(
            variant, "local_field_helmholtz_synthesis_frequency_softmax", False
        ),
        dense_local_field_helmholtz_synthesis_source_onset_phase=getattr(
            variant, "local_field_helmholtz_synthesis_source_onset_phase", False
        ),
        dense_local_field_helmholtz_source_relative_coordinates=getattr(
            variant, "local_field_helmholtz_source_relative_coordinates", False
        ),
        dense_local_field_helmholtz_spectral_bypass=getattr(
            variant, "local_field_helmholtz_spectral_bypass", False
        ),
        dense_local_field_helmholtz_spectral_bypass_per_branch=getattr(
            variant, "local_field_helmholtz_spectral_bypass_per_branch", False
        ),
        dense_local_field_helmholtz_background_conditioning=getattr(
            variant, "local_field_helmholtz_background_conditioning", False
        ),
        dense_local_field_helmholtz_background_sigma_cells=getattr(
            variant, "local_field_helmholtz_background_sigma_cells", 2.0
        ),
        dense_local_field_helmholtz_background_global_propagator=getattr(
            variant, "local_field_helmholtz_background_global_propagator", False
        ),
        dense_local_field_helmholtz_background_propagation_modes=getattr(
            variant, "local_field_helmholtz_background_propagation_modes", 48
        ),
        dense_local_field_helmholtz_background_direct_frequency_head=getattr(
            variant,
            "local_field_helmholtz_background_direct_frequency_head",
            False,
        ),
        dense_local_field_helmholtz_background_direct_frequencies=getattr(
            variant,
            "local_field_helmholtz_background_direct_frequencies",
            32,
        ),
        dense_local_field_helmholtz_background_direct_spectral_experts=getattr(
            variant,
            "local_field_helmholtz_background_direct_spectral_experts",
            0,
        ),
        dense_local_field_adapter_gate_init=getattr(
            variant, "local_field_adapter_gate_init", 0.0
        ),
        dense_high_frequency_residual=getattr(variant, "high_frequency_residual", False),
        dense_high_frequency_hidden=getattr(variant, "high_frequency_hidden", 64),
        dense_high_frequency_depth=getattr(variant, "high_frequency_depth", 3),
        use_local_phase=variant.use_local_phase, activation_checkpointing=True,
        heads=cfg.heads, token_grid=token_grid, position_bands=cfg.position_bands,
        fourier_bands=cfg.fourier_bands, gabor_scales_s=cfg.gabor_scales_s,
        ray_samples=cfg.ray_samples, domain_x_m=cfg.domain_x_m,
        domain_z_m=cfg.domain_z_m, domain_t_s=cfg.domain_t_s,
    )


def _loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    gradient_weight: float,
    spectrum_weight: float = 0.0,
    per_frame: bool = False,
    frame_energy_floor_fraction: float = 0.0,
    frame_time_weights: torch.Tensor | None = None,
):
    if per_frame:
        frame = frame_relative_l2(
            prediction, target,
            energy_floor_fraction=frame_energy_floor_fraction,
            frame_time_weights=frame_time_weights,
        )
    else:
        error = (prediction.float() - target.float()).flatten(1).norm(dim=-1)
        target_norm = target.float().flatten(1).norm(dim=-1)
        frame = (error / target_norm.clamp_min(1e-8)).mean()
    pred_dx = prediction[..., 1:] - prediction[..., :-1]
    true_dx = target[..., 1:] - target[..., :-1]
    pred_dz = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_dz = target[..., 1:, :] - target[..., :-1, :]
    gradient = (pred_dx.float() - true_dx.float()).square().mean().sqrt()
    gradient = gradient + (pred_dz.float() - true_dz.float()).square().mean().sqrt()
    scale = true_dx.float().square().mean().sqrt() + true_dz.float().square().mean().sqrt()
    gradient = gradient / scale.clamp_min(1e-8)
    spectrum = (
        band_limited_residual_loss(prediction, target)
        if spectrum_weight > 0.0
        else prediction.new_zeros(())
    )
    return frame + gradient_weight * gradient + spectrum_weight * spectrum, frame, gradient, spectrum


@torch.no_grad()
def _evaluate(model, batches, normalizer, device, microbatch_records, floor_fraction):
    model.eval(); predictions=[]; targets=[]; families=[]
    for batch in batches:
        for micro in split_pilot_batch(batch, microbatch_records=microbatch_records):
            tensors=_to_device(micro, device); source=tensors["source_parameters"]
            prepared=model.prepare_sources(model.encode_medium(tensors["velocity_mps"], normalizer),source,tensors["source_map"],normalizer,record_to_medium=tensors["record_to_medium"])
            prediction=model.dense_normalized(prepared,tensors["requested_time_s"],x_m=tensors["x_m"],z_m=tensors["z_m"],time_block=1)
            target=normalizer.encode_pressure(tensors["dense_target_physical"],source[:,4])
            predictions.append(prediction.cpu()); targets.append(target.cpu()); families.extend(micro.medium_type)
    return exact_wavefield_metrics(torch.cat(predictions),torch.cat(targets),families=tuple(families),energy_floor_fraction=floor_fraction)


def run_variant(name, variant, config, base, manifest, normalizer, *, device, smoke_updates):
    root=Path(config["artifact_dir"])/("smoke" if smoke_updates else "variants")/name; root.mkdir(parents=True,exist_ok=True)
    epochs=1 if smoke_updates else int(config["epochs"]); steps_per_epoch=1 if smoke_updates else int(config["train_macro_steps"])
    macro_accumulation=1 if smoke_updates else int(config.get("accumulate_macros",1))
    if macro_accumulation <= 0: raise ValueError("accumulate_macros must be positive")
    total_steps=epochs*steps_per_epoch; seed=int(config["seed"])
    identity={"schema":"saved_time_v4_probe_v1","variant":name,"variant_config":variant.__dict__,"manifest_digest":manifest.digest,"config":config,"smoke_updates":int(smoke_updates)}
    run_digest=_digest(identity); identity["run_digest"]=run_digest
    identity_path=root/"run_identity.json"
    if identity_path.exists() and json.load(identity_path.open()) != identity: raise ValueError(f"identity mismatch: {name}")
    if not identity_path.exists(): _atomic_json(identity,identity_path)
    terminal=root/"terminal.json"
    if terminal.exists() and json.load(terminal.open()).get("status")=="complete": return json.load(terminal.open())
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=_model(base,manifest,variant).to(device); load_v3_backbone(model,config["parent_checkpoint"])
    dense=list(model.dense_decoder.parameters()); dense_ids={id(p) for p in dense}; backbone=[p for p in model.parameters() if id(p) not in dense_ids]
    optcfg=config["optimizer"]
    optimizer=torch.optim.AdamW([{"params":dense,"lr":float(optcfg["head_learning_rate"])},{"params":backbone,"lr":float(optcfg["backbone_learning_rate"])}],weight_decay=float(optcfg["weight_decay"]))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(total_steps,1))
    start_epoch=0; global_step=0; latest=root/"latest.pt"
    if latest.exists():
        meta=load_checkpoint(latest,model=model,optimizer=optimizer,expected_manifest_digest=manifest.digest,expected_config_digest=run_digest,restore_rng=True,map_location=device); start_epoch=meta.epoch; global_step=meta.global_step
        scheduler.last_epoch=global_step
        scheduler._step_count=global_step+1
        scheduler._last_lr=[group["lr"] for group in optimizer.param_groups]
    train_schedule=_schedule(manifest,split="train",pool_steps=int(config["train_macro_steps"]),total_steps=total_steps*macro_accumulation,seed=seed)
    train_ds=ExactStoredTimeBatchDataset(base.data.source_h5,manifest,split="train",schedule=train_schedule,query_points=int(config["query_points"]),seed=seed)
    loader=make_pilot_loader(train_ds,workers=0 if smoke_updates else int(config["workers"]),prefetch_factor=int(config["prefetch_factor"]),pin_memory=True)
    val_schedule=_schedule(manifest,split="validation",pool_steps=int(config["validation_macro_steps"]),total_steps=int(config["validation_macro_steps"]),seed=seed+7919,offset=10000)
    val_ds=ExactStoredTimeBatchDataset(base.data.source_h5,manifest,split="validation",schedule=val_schedule,query_points=int(config["query_points"]),seed=seed+7919)
    val_batches=[val_ds[index] for index in range(len(val_ds))]
    iterator=iter(loader); torch.cuda.reset_peak_memory_stats(); report={}
    for epoch in range(start_epoch,epochs):
        model.train(); epoch_loss=[]
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(macro_accumulation):
                macro=next(iterator); pieces=split_pilot_batch(macro,microbatch_records=int(config["microbatch_records"]))
                for micro in pieces:
                    tensors=_to_device(micro,device); source=tensors["source_parameters"]
                    prepared=model.prepare_sources(model.encode_medium(tensors["velocity_mps"],normalizer),source,tensors["source_map"],normalizer,record_to_medium=tensors["record_to_medium"])
                    prediction=model.dense_normalized(prepared,tensors["requested_time_s"],x_m=tensors["x_m"],z_m=tensors["z_m"],time_block=1)
                    target=normalizer.encode_pressure(tensors["dense_target_physical"],source[:,4])
                    loss,frame,gradient,spectrum=_loss(prediction,target,gradient_weight=float(config["loss"]["spatial_gradient"]),spectrum_weight=float(config["loss"].get("spectrum",0.0))); (loss/(len(pieces)*macro_accumulation)).backward(); epoch_loss.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(optcfg["gradient_clip"])); optimizer.step(); scheduler.step(); global_step+=1
        metrics=_evaluate(model,val_batches,normalizer,device,int(config["microbatch_records"]),float(config["energy_floor_fraction"])); metrics["train_loss"]=float(np.mean(epoch_loss)); metrics["peak_cuda_bytes"]=float(torch.cuda.max_memory_allocated())
        checkpoint=root/"checkpoints"/f"epoch_{epoch+1:04d}.pt"; save_checkpoint_atomic(checkpoint,model=model,optimizer=optimizer,epoch=epoch+1,global_step=global_step,manifest_digest=manifest.digest,config_digest=run_digest,metrics={"validation":metrics["aggregate_floored_relative_l2"]})
        _atomic_hardlink(checkpoint,latest); report={"event":"epoch","epoch":epoch+1,"global_step":global_step,"metrics":metrics,"checkpoint":str(checkpoint)}
        with (root/"metrics.jsonl").open("a",encoding="utf8") as handle: handle.write(json.dumps(report,sort_keys=True)+"\n")
        best_path=root/"best.json"
        best=json.load(best_path.open()) if best_path.exists() else None
        if best is None or metrics["aggregate_floored_relative_l2"] < best["metrics"]["aggregate_floored_relative_l2"]:
            best={"epoch":epoch+1,"global_step":global_step,"metrics":metrics,"checkpoint":str(checkpoint)}
            _atomic_hardlink(checkpoint,root/"best.pt"); _atomic_json(best,best_path)
        print(json.dumps({"variant":name,**report},sort_keys=True),flush=True)
    best=json.load((root/"best.json").open())
    terminal_payload={"status":"complete","variant":name,"run_digest":run_digest,"parameter_count":sum(p.numel() for p in model.parameters()),"global_step":global_step,"metrics":best["metrics"],"checkpoint":best["checkpoint"],"best_epoch":best["epoch"]}; _atomic_json(terminal_payload,terminal); return terminal_payload


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--smoke-updates",type=int,default=0); args=parser.parse_args(argv)
    config=yaml.safe_load(Path(args.config).read_text()); base=V3Config.from_yaml(config["base_config"]); manifest=build_manifest(base.data.source_h5); validate_expected_counts(manifest,{"train":base.data.expected_train_records,"validation":base.data.expected_validation_records}); normalizer=load_normalizer(base,manifest.digest); device=torch.device("cuda")
    results={name:run_variant(name,variant,config,base,manifest,normalizer,device=device,smoke_updates=args.smoke_updates) for name,variant in probe_variants().items()}
    scores={name:float(value["metrics"]["aggregate_floored_relative_l2"]) for name,value in results.items()}; family={name:value["metrics"]["family_floored_relative_l2"] for name,value in results.items()}
    try: selected=select_probe_candidate(scores,family,minimum_relative_improvement=float(config["selection"]["minimum_relative_improvement"]),family_regression_tolerance=float(config["selection"]["family_regression_tolerance"]))
    except RuntimeError: selected=None
    summary={"status":"complete","selected":selected,"scores":scores,"family_scores":family,"results":results}; _atomic_json(summary,Path(config["artifact_dir"])/("smoke_summary.json" if args.smoke_updates else "probe_summary.json")); print(json.dumps(summary,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
