#!/usr/bin/env python3
"""v19: one-shot off-panel confirmation of the v18 arm checkpoints.

Prereg: results/r16_dscp_v19_offpanel_preregistration_20260827.json.
Builds fp16 exponent-shifted bundles for the 24 final_train_confirm records via
the v18 runner's own build path (imported, not copied), scores both frozen
checkpoints with the v18 scoring path verbatim, applies the frozen consistency
readout, writes one terminal.  No training, no retries."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

v16 = load("train_r16_dscp_v16", ROOT / "scripts/train_r16_dscp_v16.py")
v18 = load("run_v18_alldata_ddp4", ROOT / "scripts/run_v18_alldata_ddp4.py")

PREREG = ROOT / "results/r16_dscp_v19_offpanel_preregistration_20260827.json"
OUT = ROOT / "results/r16_dscp_v19_offpanel"


def main() -> int:
    device = torch.device("cuda:0")
    prereg = json.loads(PREREG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "terminal.json").exists():
        raise v16.V16Refusal("v19 terminal already exists (one-shot stage)")
    log = lambda m: print(f"{v16.utc_now()} {m}", flush=True)

    panel = json.loads((ROOT / "results/r16_dscp_v1/panels.json").read_text())["records"]
    ids = sorted(r["sample_id"] for r in panel if r["role"] == "final_train_confirm")
    if len(ids) != 24:
        raise v16.V16Refusal(f"expected 24 final_train_confirm records, got {len(ids)}")

    started = time.monotonic()
    bundles, skipped, bases, scales, build_s = v18.build_partition_fp16(
        device, ids, log=log)
    if skipped:
        raise v16.V16Refusal(f"off-panel record abstained: {skipped}")
    bases_device = bases.to(device).float()

    arms = {}
    for arm, want_sha in prereg["checkpoints"].items():
        ck = ROOT / f"results/r16_dscp_v18/{arm}/best.pt"
        got = v16.sha256_file(ck)
        if got != want_sha:
            raise v16.V16Refusal(f"checkpoint sha mismatch for {arm}")
        payload = torch.load(ck, map_location="cpu", weights_only=False)
        head = v16.Wide128Head().to(device).float()
        head.load_state_dict(payload["model_state"])
        rows = [v18.v18_score_bundle(head, scales, bases_device, b, device)
                for b in bundles]
        joint = sum(r["gain_iii"] for r in rows) / len(rows)
        fam = {}
        for r in rows:
            fam.setdefault(r["family"], []).append(r["gain_iii"])
        tol_ok = sum(1 for r in rows if r["gain_iii"] >= -0.01)
        worst = min(r["gain_iii"] for r in rows)
        ref = {"A_data": 0.0224, "B_data_hinge": 0.0254}[arm]
        consistent = joint >= ref and tol_ok >= 22 and worst >= -0.02
        arms[arm] = {
            "checkpoint_sha256": got, "joint_mean_gain": joint,
            "per_family_mean_gain": {k: sum(v) / len(v) for k, v in sorted(fam.items())},
            "nonworse_within_1pct": tol_ok, "worst_harm": worst,
            "consistency_threshold_joint": ref,
            "readout": "OFF_PANEL_CONSISTENT" if consistent else "OFF_PANEL_INCONSISTENT",
            "records": rows,
        }
        log(f"[{arm}] joint {joint:+.4f} (ref {ref}) tol {tol_ok}/24 "
            f"worst {worst:+.4f} -> {arms[arm]['readout']}")

    terminal = {
        "schema": "r16_dscp_v19_offpanel_terminal_v1",
        "status": "success",
        "preregistration_sha256": v16.sha256_file(PREREG),
        "arms": arms,
        "eval_records": ids,
        "truth_scope": "train/final_train_confirm only; sealed splits untouched",
        "resources": {"wall_s": time.monotonic() - started, "build_s": build_s},
        "parent_untouched": v16.sha256_file(v16.PARENT_PATH) == v16.PARENT_SHA256,
        "completed_utc": v16.utc_now(),
    }
    v16.atomic_json(terminal, OUT / "terminal.json")
    log("terminal written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
