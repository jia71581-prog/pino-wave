"""CPU warm-start de-risk for the C1 (windowed_propagation, gate_init=0.02) launch.

Mirrors the REAL launch path exactly: calls the trainer's own _load_context +
_load_parent_model against the actual warp_r1/best.pt parent on CPU. load_checkpoint
raises on ANY unexpected key / non-allow-listed missing key / shape drift / digest
mismatch, so a clean return == the warm-start contract holds. Then asserts the C1
single-factor invariants (gate == gate_init, param delta == windowed_propagation only).
"""
from __future__ import annotations

import json
import sys
import dataclasses
from pathlib import Path

import yaml
import torch

sys.argv = ["cpu-check"]
import importlib.util

spec = importlib.util.spec_from_file_location(
    "train_v4", "scripts/train_saved_time_v4_full_support.py"
)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

CFG = "configs/saved_time_v4/generated/local_field_w128_windowed_propagation_c1.yaml"
config = yaml.safe_load(Path(CFG).read_text())

base, manifest, parent_identity = train._load_context(config)
print(f"[ctx] manifest.digest={manifest.digest[:12]}...  parent_identity ok")

child_variant = train.probe_variant_for_config(config, parent_identity)
print(
    f"[variant] windowed_propagation={child_variant.local_field_windowed_propagation} "
    f"window={child_variant.local_field_windowed_propagation_window} "
    f"rank={child_variant.local_field_windowed_propagation_rank} "
    f"gate_init={child_variant.local_field_adapter_gate_init} "
    f"warp={child_variant.local_field_warp}"
)
assert child_variant.local_field_windowed_propagation is True
assert child_variant.local_field_windowed_propagation_window == 2
assert abs(child_variant.local_field_adapter_gate_init - 0.02) < 1e-9
assert child_variant.local_field_warp is True  # C1 advects the warp-corrected field

# THE launch-critical call: real warp_r1 -> C1 child warm-start on CPU.
model = train._load_parent_model(config, base, manifest, parent_identity, device="cpu")
print("[warmstart] _load_parent_model returned cleanly (no unexpected/forbidden-missing/shape/digest error)")

wp = model.local_field.windowed_propagation
assert wp is not None, "windowed_propagation module missing after build"
g = wp.gate.detach()
print(f"[gate] gate = {float(g)}")
assert torch.allclose(g, torch.tensor(0.02), atol=1e-6), g

# single-factor param accounting vs the warp-only parent
warp_only = dataclasses.replace(
    child_variant,
    local_field_windowed_propagation=False,
    local_field_adapter_gate_init=0.0,
)
parent_model = train._model(base, manifest, warp_only)
n_child = sum(p.numel() for p in model.parameters())
n_parent = sum(p.numel() for p in parent_model.parameters())
n_wp = sum(p.numel() for p in wp.parameters())
print(f"[params] child={n_child}  warp_only={n_parent}  windowed_propagation={n_wp}  delta={n_child-n_parent}")
assert n_child - n_parent == n_wp, "NOT single-factor: extra params beyond windowed_propagation"

bad = [n for n, p in model.named_parameters()
       if any(p is q for q in wp.parameters()) and not n.startswith("local_field.windowed_propagation.")]
assert not bad, f"C1 params not under local_field.windowed_propagation. prefix: {bad}"

print("\nC1 WARM-START CONTRACT: PASS")
print(f"  - real warp_r1/best.pt loaded into C1 child with zero contract violations")
print(f"  - gate warmed to 0.02 (deadlock-fixed; center tap keeps gate->0 an exact no-op)")
print(f"  - single factor: +{n_wp} params, all under local_field.windowed_propagation.")
