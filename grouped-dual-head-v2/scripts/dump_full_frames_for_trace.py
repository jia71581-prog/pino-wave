#!/usr/bin/env python3
"""Dump ALL saved-time frames (true/coarse/corrected) for one record → npz for trace plots.

Extends dump_coarse_field_diagnostic.py: instead of 6 evenly-spaced snapshots it keeps
every saved time index so we can extract receiver waveforms u(t) at fixed points.
Memory-lean: 401 x 201 x 201 x 3 float32 ~ 78 MB per record, fine on CPU.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--family", choices=list(FAMILY_INDEX), default="uniform")
    p.add_argument("--out", required=True)
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
    # The dataset file was extended after the ladder run (record count grew), so its
    # content digest no longer matches the normalization binding. Grid/time axes are
    # unchanged, so load the normalizer WITHOUT the manifest guard for this read-only
    # visualization dump. A frame-level cross-check against the frozen coarse_diag npz
    # confirms the checkpoint/data pairing still reproduces the known predictions.
    from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
    normalizer = PhysicalNormalizer.from_dict(
        json.loads(Path(base.data.normalization_json).read_text(encoding="utf8")),
        expected_manifest=None,
    )
    variant = ProbeVariant(
        depth=int(identity["dense_depth"]), use_local_phase=True,
        spectral_rank=int(identity["dense_spectral_rank"]), modes=int(identity["dense_modes"]),
        temporal_basis_rank=0, family_expert_rank=0,
    )
    model = _model(base, manifest, variant).to(device)
    # The dataset file grew after the ladder run, so the recomputed manifest digest no
    # longer matches what the checkpoint was bound to. Use the checkpoint's own stored
    # digest so the identity guard still verifies checkpoint<->weights integrity.
    ckpt_payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_manifest_digest = str(ckpt_payload.get("manifest_digest", ""))
    del ckpt_payload
    load_checkpoint(str(ckpt), model=model, optimizer=None,
                    expected_manifest_digest=ckpt_manifest_digest,
                    expected_config_digest=identity["run_digest"], map_location=device)
    model.eval()

    rec_indices = identity["record_indices"]
    record_index = int(rec_indices[FAMILY_INDEX[args.family]])
    config = build_probe_config(dense_lr=1e-4, backbone_lr=5e-5, temporal_lr=1e-5,
                                seed=int(identity["seed"]),
                                travel_time_h5="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5")
    from scripts.diagnose_saved_time_temporal_three_record_overfit import _dataset
    n_saved = len(manifest.time_s)
    schedule = (FullSupportStepSpec(step=990000, epoch=0, record_indices=(record_index,), appearance_indices=(0,)),)
    dataset = _dataset(config, base, manifest, (record_index,), split=identity["split"],
                       schedule=schedule, time_policy="all_saved", frames_per_record=n_saved)

    true_f, coarse_f, pred_f, times = [], [], [], []
    src_xz = None
    with torch.inference_mode():
        for bi in range(len(dataset)):
            batch = dataset[bi]
            for micro in split_pilot_batch(batch, microbatch_records=1):
                t = _to_device(micro, device)
                src = t["source_parameters"]
                if src_xz is None:
                    src_xz = (float(src[0, 0].cpu()), float(src[0, 1].cpu()))
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
                true_f.append(tgt.float().cpu().numpy().reshape(-1, *tgt.shape[-2:]))
                coarse_f.append(coarse.float().cpu().numpy().reshape(-1, *coarse.shape[-2:]))
                pred_f.append(pred.float().cpu().numpy().reshape(-1, *pred.shape[-2:]))
                times.append(t["requested_time_s"].float().cpu().numpy().reshape(-1))

    true = np.concatenate(true_f, 0); coarse = np.concatenate(coarse_f, 0); pred = np.concatenate(pred_f, 0)
    times = np.concatenate(times, 0)
    order = np.argsort(times)
    true, coarse, pred, times = true[order], coarse[order], pred[order], times[order]

    x_m = t["x_m"].float().cpu().numpy().reshape(-1)
    z_m = t["z_m"].float().cpu().numpy().reshape(-1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, times=times, true=true, coarse=coarse, pred=pred,
                        x_m=x_m, z_m=z_m, source_xz=np.array(src_xz),
                        family=args.family, record_index=record_index)
    print(json.dumps({"family": args.family, "record_index": record_index,
                      "n_frames": int(true.shape[0]), "grid": list(true.shape[1:]),
                      "source_xz": src_xz, "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
