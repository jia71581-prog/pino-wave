"""Contract check for the JOINT spectral-bypass + late-rank continue-pretraining warm start.

VERDICT sec 32 left the joint move untested: the spectral bypass lifted the render's
effective rank 8 -> 14, but the rank-8 synthesis head truncates the high-rank late
amplitude the bypass now makes available.  The fix is to warm-start from the spectral-injection
checkpoint (bypass weights TRANSFER by shape, render stays rank ~14) and add the fresh
zero-init late-rank head (higher-rank synthesis confined to the late-dominant low-freq bins).

This script asserts the continue-pretraining contract holds for that combination:
  1. every spectral-bypass tensor LOADS from the checkpoint (is NOT counted fresh),
  2. only the late_* tensors are fresh,
  3. at step 0 the warm-started model reproduces the parent's local-field output bit-for-bit
     (zero-init late mixing => no-op), on a random but fixed synthetic render.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import dataclasses

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.probe import ProbeVariant
from scripts.train_saved_time_v4_probe import _model
from scripts.diagnose_capacity_ladder_overfit import build_base_config
from scripts.diagnose_helmholtz_g3_heldout import warmstart_full_helmholtz


CKPT = "results/helmholtz_g3_specinj_gate01_N192_ddp4/checkpoints/update_0240.pt"
NORM = "grouped_v3_normalization/before_tgrs_ablation_identity_20260726T1050.json"


def build(variant_kwargs):
    base = build_base_config(128)
    data_cfg = dataclasses.replace(base.data, normalization_json=NORM)
    base = dataclasses.replace(base, data=data_cfg)
    manifest = build_manifest(base.data.source_h5)
    variant = ProbeVariant(
        depth=8,
        use_local_phase=True,
        spectral_rank=112,
        modes=32,
        temporal_basis_rank=0,
        family_expert_rank=0,
        local_field=True,
        local_field_residual=False,
        local_field_helmholtz_synthesis=True,
        local_field_helmholtz_synthesis_frequencies=64,
        local_field_helmholtz_synthesis_wkb_phase=True,
        local_field_helmholtz_synthesis_rank=8,
        **variant_kwargs,
    )
    return _model(base, manifest, variant)


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)

    # Target model: rank-8 base synthesis + fresh late-rank 32 head + spectral bypass.
    model = build(dict(
        local_field_helmholtz_synthesis_late_rank=32,
        local_field_helmholtz_synthesis_late_frequencies=24,
        local_field_helmholtz_spectral_bypass=True,
    ))
    report = warmstart_full_helmholtz(model, Path(CKPT))
    print("warmstart report:", report)

    # ---- 1 & 2: which keys came from the checkpoint vs stayed fresh -------------
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    src = payload["model_state"] if "model_state" in payload else payload
    tgt_keys = set(model.state_dict().keys())
    src_keys = set(src.keys())

    bypass_tgt = {k for k in tgt_keys if "spectral_bypass" in k}
    bypass_loaded = {k for k in bypass_tgt if k in src_keys}
    late_tgt = {k for k in tgt_keys if ".late_" in k}
    late_in_src = {k for k in late_tgt if k in src_keys}

    print(f"spectral-bypass tensors: {len(bypass_tgt)} in model, {len(bypass_loaded)} loaded from ckpt")
    print(f"late tensors: {len(late_tgt)} in model, {len(late_in_src)} present in ckpt (want 0)")

    assert bypass_tgt, "no spectral-bypass tensors in target model"
    assert bypass_loaded == bypass_tgt, (
        f"spectral-bypass NOT fully loaded (fresh: {sorted(bypass_tgt - bypass_loaded)})")
    assert late_tgt, "no late-rank tensors in target model"
    assert not late_in_src, f"late tensors unexpectedly in ckpt: {sorted(late_in_src)}"

    # ---- 3: step-0 bit-for-bit reproduction of the parent local field ----------
    parent = build(dict(
        local_field_helmholtz_synthesis_late_rank=0,
        local_field_helmholtz_synthesis_late_frequencies=0,
        local_field_helmholtz_spectral_bypass=True,
    ))
    p_report = warmstart_full_helmholtz(parent, Path(CKPT))
    print("parent warmstart report:", p_report)

    lf_child = model.local_field.helmholtz_synthesis
    lf_parent = parent.local_field.helmholtz_synthesis
    lf_child.eval(); lf_parent.eval()

    records, width, H, W = 2, 128, 51, 51
    rendered = torch.randn(records, width, H, W)
    arrival = torch.rand(records, H, W) * 0.5
    count = 8
    time_s = torch.rand(records, count) * 1.0
    saved = torch.linspace(0.0, 1.0, 401)
    with torch.no_grad():
        out_child = lf_child(rendered, arrival, time_s, saved, domain_t_s=1.0)
        out_parent = lf_parent(rendered, arrival, time_s, saved, domain_t_s=1.0)
    max_abs = (out_child - out_parent).abs().max().item()
    print(f"step-0 max|child - parent| synthesis output = {max_abs:.3e}")
    assert max_abs == 0.0, f"late head not a no-op at init: max diff {max_abs}"

    print("\nALL CONTRACT CHECKS PASSED: spectral-bypass loaded, late head fresh & zero-init no-op.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
