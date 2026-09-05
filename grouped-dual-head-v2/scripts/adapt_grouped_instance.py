#!/usr/bin/env python
"""Causal two-onset-frame adaptation for one grouped single-source record."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import h5py, torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from grouped_ufno_mionet import OperatorConfig, GroupedSingleSourceUFNOMIONetOperator
from grouped_ufno_mionet.instance_adaptation.data import OnsetDeploymentDataset
from grouped_ufno_mionet.instance_adaptation.model import OnsetAdaptedOperator
from grouped_ufno_mionet.instance_adaptation.trainer import adapt_instance
from grouped_ufno_mionet.training.checkpoint import load_checkpoint

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--sample-index", type=int, required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--config", default="grouped_ufno_mionet/configs/production.yaml")
    p.add_argument("--steps", type=int, default=20); p.add_argument("--device", default="cuda")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv); device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    base = GroupedSingleSourceUFNOMIONetOperator(OperatorConfig.from_yaml(args.config)).to(device)
    load_checkpoint(args.checkpoint, base, map_location=device)
    example = OnsetDeploymentDataset(args.dataset)[args.sample_index]
    model = OnsetAdaptedOperator(base).to(device)
    velocity, source, time_s, observed = (example.velocity_mps[None, None].to(device), example.source[None].to(device), example.time_s.to(device), example.observed_wavefield[None].to(device))
    report = adapt_instance(model, velocity=velocity, source=source, time_s=time_s, observed_wavefield=observed, observed_indices=example.observed_indices, steps=args.steps)
    field = model(velocity, source, time_s, observed, example.observed_indices).detach().cpu()[0]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "report": report, "observed_indices": example.observed_indices}, out / "adapted.pt")
    with h5py.File(out / "wavefield.h5", "w") as h5:
        h5["pressure"] = field.numpy(); h5["time_s"] = example.time_s.numpy(); h5.attrs["observed_indices"] = example.observed_indices
    (out / "metrics.json").write_text(json.dumps({k:v for k,v in report.items() if k not in {"state_dict", "baseline_state_dict"}}, indent=2))
if __name__ == "__main__": main()
