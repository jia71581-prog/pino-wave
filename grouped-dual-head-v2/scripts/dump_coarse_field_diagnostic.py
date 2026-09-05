#!/usr/bin/env python3
"""Dump true / coarse-MIONet / corrected wavefields for one record from a capacity-ladder checkpoint.

Diagnoses *where* the coarse MIONet field fails (the bottleneck identified by the
capacity ladder): dumps normalized pressure fields at a set of saved times plus a
per-saved-time relative-L2 curve, for true, coarse (MIONet), and corrected
(coarse+decoder) predictions. Output .npz feeds the wavefield-viz skill.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from saved_time_phase_operator_v4.probe import ProbeVariant
from saved_time_phase_operator_v4.data import split_pilot_batch
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec
from saved_time_phase_operator_v4.losses import apply_hard_causality, source_causality_onset_s

from scripts.train_saved_time_v4_probe import _model
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.diagnose_capacity_ladder_overfit import build_base_config, build_probe_config


FAMILY_INDEX = {"uniform": 0, "layered": 1, "marmousi": 2}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="capacity-ladder run dir with run_identity.json + best checkpoint")
    p.add_argument("--checkpoint", help="override checkpoint path (default: best from terminal.json)")
    p.add_argument("--family", choices=list(FAMILY_INDEX), default="uniform")
    p.add_argument("--out", required=True)
    p.add_argument("--num-dump-frames", type=int, default=6, help="evenly spaced saved-time snapshots to dump as fields")
    args = p.parse_args(argv)

    run = Path(args.run_dir)
    identity = json.loads((run / "run_identity.json").read_text())
    ckpt = args.checkpoint
    if ckpt is None:
        term = json.loads((run / "terminal.json").read_text())
        ckpt = term.get("best_checkpoint") or str(run / "best.pt")
    ckpt = Path(ckpt)

    device = torch.device("cuda")
    base = build_base_config(int(identity["width"]))
    manifest = build_manifest(base.data.source_h5)
    normalizer = load_normalizer(base, manifest.digest)
    variant = ProbeVariant(
        depth=int(identity["dense_depth"]), use_local_phase=True,
        spectral_rank=int(identity["dense_spectral_rank"]), modes=int(identity["dense_modes"]),
        temporal_basis_rank=0, family_expert_rank=0,
    )
    model = _model(base, manifest, variant).to(device)
    load_checkpoint(
        str(ckpt), model=model, optimizer=None,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=identity["run_digest"], map_location=device,
    )
    model.eval()

    # pick the record index for the requested family (from run_identity's stored triplet)
    rec_indices = identity["record_indices"]
    fam_i = FAMILY_INDEX[args.family]
    record_index = int(rec_indices[fam_i])

    config = build_probe_config(dense_lr=1e-4, backbone_lr=5e-5, temporal_lr=1e-5,
                                seed=int(identity["seed"]),
                                travel_time_h5="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5")
    from scripts.diagnose_saved_time_temporal_three_record_overfit import _dataset
    n_saved = len(manifest.time_s)
    schedule = (FullSupportStepSpec(step=990000, epoch=0, record_indices=(record_index,), appearance_indices=(0,)),)
    dataset = _dataset(config, base, manifest, (record_index,), split=identity["split"],
                       schedule=schedule, time_policy="all_saved", frames_per_record=n_saved)

    true_frames, coarse_frames, pred_frames, times = [], [], [], []
    with torch.inference_mode():
        for bi in range(len(dataset)):
            batch = dataset[bi]
            for micro in split_pilot_batch(batch, microbatch_records=1):
                t = _to_device(micro, device)
                src = t["source_parameters"]
                prepared = model.prepare_sources(model.encode_medium(t["velocity_mps"], normalizer),
                                                 src, t["source_map"], normalizer, record_to_medium=t["record_to_medium"])
                dg = model.prepare_dense_grid(prepared, x_m=t["x_m"], z_m=t["z_m"],
                                              travel_time_s=None if micro.dense_travel_time_s is None else micro.dense_travel_time_s.to(device))
                pred, coarse = model.dense_normalized_with_coarse(prepared, t["requested_time_s"], dense_grid=dg, time_block=1)
                tgt = normalizer.encode_pressure(t["dense_target_physical"], src[:, 4])
                if bool(config["loss"].get("hard_causality", False)):
                    onset = source_causality_onset_s(src, lead_cycles=float(config["loss"].get("hard_causality_lead_cycles", 0.0)))
                    pred = apply_hard_causality(pred, t["requested_time_s"], onset)
                    coarse = apply_hard_causality(coarse, t["requested_time_s"], onset)
                true_frames.append(tgt.float().cpu().numpy().reshape(-1, *tgt.shape[-2:]))
                coarse_frames.append(coarse.float().cpu().numpy().reshape(-1, *coarse.shape[-2:]))
                pred_frames.append(pred.float().cpu().numpy().reshape(-1, *pred.shape[-2:]))
                times.append(t["requested_time_s"].float().cpu().numpy().reshape(-1))

    true = np.concatenate(true_frames, 0); coarse = np.concatenate(coarse_frames, 0); pred = np.concatenate(pred_frames, 0)
    times = np.concatenate(times, 0)
    order = np.argsort(times); true, coarse, pred, times = true[order], coarse[order], pred[order], times[order]

    # per-saved-time relative L2 (frame-normalized)
    def per_t_rl2(a):
        num = np.linalg.norm((a - true).reshape(a.shape[0], -1), axis=1)
        den = np.linalg.norm(true.reshape(true.shape[0], -1), axis=1)
        return num / np.clip(den, 1e-8, None)
    rl2_coarse, rl2_pred = per_t_rl2(coarse), per_t_rl2(pred)

    dump_idx = np.linspace(0, len(times) - 1, int(args.num_dump_frames)).round().astype(int)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        times=times, dump_idx=dump_idx,
                        true=true[dump_idx], coarse=coarse[dump_idx], pred=pred[dump_idx],
                        rl2_coarse=rl2_coarse, rl2_pred=rl2_pred, family=args.family)
    print(json.dumps({
        "family": args.family, "record_index": record_index, "n_saved": int(len(times)),
        "coarse_mean_rl2": float(rl2_coarse.mean()), "pred_mean_rl2": float(rl2_pred.mean()),
        "coarse_late_rl2": float(rl2_coarse[len(times)//2:].mean()), "pred_late_rl2": float(rl2_pred[len(times)//2:].mean()),
        "dump_times_s": [round(float(times[i]), 4) for i in dump_idx], "out": args.out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
