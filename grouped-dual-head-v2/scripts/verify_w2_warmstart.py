#!/usr/bin/env python3
"""Pre-flight: confirm the W2 full warm start reproduces W2 at update-0.

Read-only w.r.t. the artifact tree — writes nothing. Builds a width=128 /
dense_depth=12 SavedTimePhaseOperatorV4 with a zero-init local_field residual,
loads every shape-matching tensor from the W2_w128_d12 update_0800 checkpoint,
prints the loaded/fresh/unexpected key audit, then runs one validation_fixed
triplet evaluation. Because the residual output conv is zero-init, the baseline
metric must reproduce W2's trained fixed metric (agg approximately 0.13736,
families layered approximately 0.1168 / marmousi approximately 0.1621 /
uniform approximately 0.1333).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.probe import ProbeVariant
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts

from scripts.train_saved_time_v4_probe import _model
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.diagnose_capacity_ladder_overfit import (
    build_base_config,
    build_probe_config,
    warmstart_full_checkpoint,
)
from scripts.diagnose_saved_time_temporal_three_record_overfit import (
    select_one_index_per_family,
    _evaluate_triplet,
)

W2_CKPT = (
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "capacity_ladder/W2_w128_d12/checkpoints/update_0800.pt"
)
W2_FIXED_AGG = 0.13736091246160595
W2_FIXED_FAMILY = {"layered": 0.116759054684129, "marmousi": 0.16205638200199907, "uniform": 0.13326730069868978}


def main() -> int:
    seed = 372
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device("cuda")

    base = build_base_config(128)
    manifest = build_manifest(base.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": base.data.expected_train_records,
            "validation": base.data.expected_validation_records,
        },
    )
    normalizer = load_normalizer(base, manifest.digest)

    variant = ProbeVariant(
        depth=12,
        use_local_phase=True,
        spectral_rank=112,
        modes=32,
        local_field=True,
        local_field_channel_multipliers=(1, 1, 2, 2),
        local_field_causal_width_s=0.005,
        local_field_residual=True,
    )
    model = _model(base, manifest, variant).to(device)

    report = warmstart_full_checkpoint(model, Path(W2_CKPT))
    print("=== WARMSTART AUDIT ===")
    print(json.dumps({k: v for k, v in report.items() if k not in ("loaded_keys_sample",)}, indent=2, sort_keys=True))
    # every fresh key must be a local_field tensor
    assert all(k.startswith("local_field.") for k in report["fresh_keys"]), report["fresh_keys"]
    assert not report["unexpected_source_keys"], report["unexpected_source_keys"]
    assert not report["shape_mismatches"], report["shape_mismatches"]

    config = build_probe_config(
        dense_lr=1.0e-4, backbone_lr=5.0e-5, temporal_lr=1.0e-5, seed=seed,
        travel_time_h5="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5",
    )
    indices = select_one_index_per_family(manifest.records, split="train")

    baseline = _evaluate_triplet(
        model, base, manifest, normalizer, device, config, indices,
        split="train", time_policy="validation_fixed", frames_per_record=32,
    )
    agg = float(baseline["aggregate_relative_l2"])
    fam = {k: float(v) for k, v in baseline["family_relative_l2"].items()}
    print("=== UPDATE-0 BASELINE (validation_fixed, 32 frames) ===")
    print(json.dumps({"aggregate_relative_l2": agg, "family_relative_l2": fam}, indent=2, sort_keys=True))
    print("=== W2 REFERENCE (update_0800 fixed) ===")
    print(json.dumps({"aggregate_relative_l2": W2_FIXED_AGG, "family_relative_l2": W2_FIXED_FAMILY}, indent=2, sort_keys=True))

    agg_err = abs(agg - W2_FIXED_AGG)
    fam_err = {k: abs(fam[k] - W2_FIXED_FAMILY[k]) for k in W2_FIXED_FAMILY}
    max_err = max([agg_err, *fam_err.values()])
    print(f"=== MAX ABS ERROR vs W2: {max_err:.3e} (agg {agg_err:.3e}, family {fam_err}) ===")
    reproduces = max_err < 1.0e-3
    print("REPRODUCES_W2:", reproduces)
    return 0 if reproduces else 2


if __name__ == "__main__":
    raise SystemExit(main())
