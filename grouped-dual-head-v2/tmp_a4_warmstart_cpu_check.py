"""CPU warm-start de-risk for the A4 (multi_arrival, gate_init=0.03) launch.

Mirrors the REAL launch path exactly: it calls the trainer's own _load_context +
_load_parent_model against the actual warp_r1/best.pt parent on CPU. load_checkpoint
raises on ANY unexpected key / non-allow-listed missing key / shape drift / digest
mismatch, so a clean return == the warm-start contract holds. We then assert the A4
single-factor invariants (path_gate == gate_init, params delta == multi_arrival only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
import torch

sys.argv = ["cpu-check"]  # guard against arg parsing at import
import importlib.util

spec = importlib.util.spec_from_file_location(
    "train_v4", "scripts/train_saved_time_v4_full_support.py"
)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

CFG = "configs/saved_time_v4/generated/local_field_w128_multi_arrival_a4.yaml"
config = yaml.safe_load(Path(CFG).read_text())

base, manifest, parent_identity = train._load_context(config)
print(f"[ctx] manifest.digest={manifest.digest[:12]}...  parent_identity ok")

# candidate (A4 child) variant straight from the config
child_variant = train.probe_variant_for_config(config, parent_identity)
print(
    f"[variant] multi_arrival={child_variant.local_field_multi_arrival} "
    f"paths={child_variant.local_field_multi_arrival_paths} "
    f"gate_init={child_variant.local_field_adapter_gate_init} "
    f"warp={child_variant.local_field_warp}"
)
assert child_variant.local_field_multi_arrival is True
assert child_variant.local_field_multi_arrival_paths == 3
assert abs(child_variant.local_field_adapter_gate_init - 0.03) < 1e-9
assert child_variant.local_field_warp is True  # Option B: path 0 stays the parent warp

# THE launch-critical call: real warp_r1 -> A4 child warm-start on CPU.
model = train._load_parent_model(config, base, manifest, parent_identity, device="cpu")
print("[warmstart] _load_parent_model returned cleanly (no unexpected/forbidden-missing/shape/digest error)")

ma = model.local_field.multi_arrival
assert ma is not None, "multi_arrival module missing after build"
pg = ma.path_gate.detach()
print(f"[gate] path_gate = {pg.tolist()}")
assert torch.allclose(pg, torch.full_like(pg, 0.03), atol=1e-6), pg
assert pg.numel() == 3

# single-factor param accounting vs the warp-only parent
parent_variant = train.probe_variant_for_config(
    json.loads(Path(str(config["parent_checkpoint_identity"])).read_text())
    if config.get("parent_checkpoint_identity") else config,
    parent_identity,
)
# build the parent (warp-only) model structurally for the param delta
from saved_time_phase_operator_v4.probe import ProbeVariant
import dataclasses

warp_only = dataclasses.replace(
    child_variant,
    local_field_multi_arrival=False,
    local_field_adapter_gate_init=0.0,
)
parent_model = train._model(base, manifest, warp_only)
n_child = sum(p.numel() for p in model.parameters())
n_parent = sum(p.numel() for p in parent_model.parameters())
n_ma = sum(p.numel() for p in ma.parameters())
print(f"[params] child={n_child}  warp_only={n_parent}  multi_arrival={n_ma}  delta={n_child-n_parent}")
assert n_child - n_parent == n_ma, "NOT single-factor: extra params beyond multi_arrival"

# every A4 param must be prefixed for LR routing
bad = [n for n, p in model.named_parameters()
       if any(p is q for q in ma.parameters()) and not n.startswith("local_field.multi_arrival.")]
assert not bad, f"A4 params not under local_field.multi_arrival. prefix: {bad}"

print("\nA4 WARM-START CONTRACT: PASS")
print(f"  - real warp_r1/best.pt loaded into A4 child with zero contract violations")
print(f"  - path_gate warmed to 0.03 (deadlock-fixed, not byte-exact by design)")
print(f"  - single factor: +{n_ma} params, all under local_field.multi_arrival.")
