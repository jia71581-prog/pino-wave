#!/usr/bin/env python3
"""Generate a formal-trainer parent_identity for the W2_w128_d12 checkpoint so the
full-support trainer can warm-start from it and add a zero-init local_field residual.

W2 is a real width-128 dense_depth-12 checkpoint (MIONet front-end + dense decoder,
no local_field). We hand it to train_saved_time_v4_full_support.py as the parent,
with the run config's variant_overrides adding local_field + local_field_residual.
local_field_missing_prefixes() then allow-lists the fresh local_field.* tensors.

Writes ONLY a new parent_identity json (never touches W2's own run_identity.json),
then dry-runs the trainer's variant/transfer resolution to prove it validates.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.probe import ProbeVariant

W2_CKPT_CONFIG_DIGEST = "6dcce8d5a46db8c7c1720a41f64ea536183f630dce921e2d7271efadc1cabb32"
W2_MANIFEST_DIGEST = "a20c9a65abbc65294062af443e2ceae241ead66450f940f652e2d95aaa0aa92b"
BASE_CONFIG = "configs/grouped_v3/continuous_pilot_w128.yaml"
OUT = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/"
    "local_field_w128_residual/w2_parent_identity.json"
)


def main() -> int:
    # W2's architecture: width comes from the base config; the variant carries the
    # dense-decoder shape (depth 12, spectral rank 112, dense modes 32) and NO
    # local_field / additive heads.
    w2_variant = ProbeVariant(
        depth=12,
        use_local_phase=True,
        spectral_rank=112,
        modes=32,
        temporal_basis_rank=0,
        family_expert_rank=0,
        band_adapter_rank=0,
        local_field=False,
        local_field_residual=False,
    )
    variant_config = dataclasses.asdict(w2_variant)

    parent_identity = {
        "schema": "local_field_warmstart_parent_v1",
        "config": {"base_config": BASE_CONFIG, "variant_overrides": {}},
        "variant_config": variant_config,
        "manifest_digest": W2_MANIFEST_DIGEST,
        "run_digest": W2_CKPT_CONFIG_DIGEST,
        "provenance": {
            "parent_checkpoint": (
                "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
                "capacity_ladder/W2_w128_d12/checkpoints/update_0800.pt"
            ),
            "note": "W2 3-record-overfit checkpoint reused as a width-128 warm-start "
            "front-end+decoder init for full-dataset local_field residual training",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(parent_identity, indent=2, sort_keys=True) + "\n")
    print("WROTE", OUT)

    # Dry-run the trainer's own resolution to prove the parent validates.
    from scripts.train_saved_time_v4_full_support import (
        probe_variant_for_config,
        resolve_spectral_mode_transfer,
        local_field_missing_prefixes,
        band_adapter_missing_prefixes,
        temporal_basis_missing_prefixes,
        family_expert_missing_prefixes,
    )

    run_config = {
        "checkpoint_transfer": {
            "allow_parent_manifest_mismatch": False,
            "parent_optimizer_state": False,
            "allow_new_local_field_parameters": True,
        },
        "variant_overrides": {"local_field": True, "local_field_residual": True},
    }
    checkpoint_identity = parent_identity  # no separate parent_checkpoint_identity
    parent_variant, candidate_variant, expands = resolve_spectral_mode_transfer(
        run_config, parent_identity=parent_identity, checkpoint_identity=checkpoint_identity
    )
    print("parent.local_field:", parent_variant.local_field, "| candidate.local_field:", candidate_variant.local_field,
          "| candidate.residual:", candidate_variant.local_field_residual, "| expand_modes:", expands)
    lf = local_field_missing_prefixes(run_config, parent_variant=parent_variant, candidate_variant=candidate_variant)
    ba = band_adapter_missing_prefixes(run_config, parent_variant=parent_variant, candidate_variant=candidate_variant)
    tb = temporal_basis_missing_prefixes(run_config, parent_variant=parent_variant, candidate_variant=candidate_variant)
    fe = family_expert_missing_prefixes(run_config, parent_variant=parent_variant, candidate_variant=candidate_variant)
    print("allowed_missing: local_field=", lf, "band_adapter=", ba, "temporal=", tb, "expert=", fe)
    assert lf == ("local_field.",), lf
    assert candidate_variant.local_field and candidate_variant.local_field_residual
    assert not expands
    print("PARENT_IDENTITY_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
