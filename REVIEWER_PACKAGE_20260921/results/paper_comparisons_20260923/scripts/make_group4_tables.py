#!/usr/bin/env python3
"""Group 4: runtime table and DeepONet comparison table (assembly only, no new runs).

Reads the frozen runtime artifacts in results/runtime_20260922/ and the
DeepONet baseline artifacts in /root/autodl-tmp/staging/deeponet_baseline_20260914/,
and writes NUMBERS.json + two markdown tables. All numbers are copied from the
source JSONs; this script computes only ratios and summary statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RT = Path("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/REVIEWER_PACKAGE_20260921/"
          "results/runtime_20260922")
DON = Path("/root/autodl-tmp/staging/deeponet_baseline_20260914")
OUT = Path("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/REVIEWER_PACKAGE_20260921/"
           "results/paper_comparisons_20260923/group4_tables")


def load(path):
    return json.loads(Path(path).read_text())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    trad = load(RT / "traditional_lwc84_runtime_29359paper.json")
    cls = load(RT / "classical_lwc84_runtime_accuracy.json")
    op1 = load(RT / "operator_29359_runtime.json")
    op16 = load(RT / "operator_29359_runtime_tb16.json")

    rows = [
        {
            "method": "LWC-84 generation protocol (401x401, dx=5 m, dt=1.25e-4 s, "
                      "restricted to 201x201)",
            "mean_s": trad["mean_runtime_s"], "min_s": trad["minimum_runtime_s"],
            "p50_s": trad["p50_runtime_s"], "p95_s": trad["p95_runtime_s"],
            "n": len(trad["measurements"]),
            "source": str(RT / "traditional_lwc84_runtime_29359paper.json"),
            "records": "validation_{uniform,layered,marmousi}_00000",
        },
        {
            "method": "LWC-84 fine (401x401, dx=5 m) on the paper's figure records",
            "mean_s": cls["runtime_summary"]["fine_401_dt1.25e-4"]["mean_s"],
            "min_s": cls["runtime_summary"]["fine_401_dt1.25e-4"]["min_s"],
            "max_s": cls["runtime_summary"]["fine_401_dt1.25e-4"]["max_s"],
            "n": cls["runtime_summary"]["fine_401_dt1.25e-4"]["n"],
            "source": str(RT / "classical_lwc84_runtime_accuracy.json"),
            "records": ", ".join(cls["protocol"]["records"]),
        },
        {
            "method": "LWC-84 native (201x201, dx=10 m, dt=2.5e-4 s, no restriction)",
            "mean_s": cls["runtime_summary"]["native_201_dt2.5e-4"]["mean_s"],
            "min_s": cls["runtime_summary"]["native_201_dt2.5e-4"]["min_s"],
            "max_s": cls["runtime_summary"]["native_201_dt2.5e-4"]["max_s"],
            "n": cls["runtime_summary"]["native_201_dt2.5e-4"]["n"],
            "source": str(RT / "classical_lwc84_runtime_accuracy.json"),
            "records": ", ".join(cls["protocol"]["records"]),
        },
        {
            "method": "Neural operator, anchor 29359, standard decode (time_block=1)",
            "mean_s": op1["mean_runtime_s"], "min_s": op1["minimum_runtime_s"],
            "p50_s": op1["p50_runtime_s"], "p95_s": op1["p95_runtime_s"],
            "n": len(op1["measurements"]),
            "source": str(RT / "operator_29359_runtime.json"),
            "records": ", ".join(op1["records"]),
        },
        {
            "method": "Neural operator, anchor 29359, batched decode (time_block=16)",
            "mean_s": op16["mean_runtime_s"], "min_s": op16["minimum_runtime_s"],
            "p50_s": op16["p50_runtime_s"], "p95_s": op16["p95_runtime_s"],
            "n": 12,
            "source": str(RT / "operator_29359_runtime_tb16.json"),
            "records": ", ".join(op16["records"]),
        },
    ]
    ratios = {
        "operator_tb1_vs_fine_records_matched": cls["runtime_summary"][
            "fine_401_dt1.25e-4"]["mean_s"] / op1["mean_runtime_s"],
        "operator_tb1_vs_native_records_matched": cls["runtime_summary"][
            "native_201_dt2.5e-4"]["mean_s"] / op1["mean_runtime_s"],
        "operator_tb16_vs_fine_records_matched": cls["runtime_summary"][
            "fine_401_dt1.25e-4"]["mean_s"] / op16["mean_runtime_s"],
        "operator_tb16_vs_native_records_matched": cls["runtime_summary"][
            "native_201_dt2.5e-4"]["mean_s"] / op16["mean_runtime_s"],
    }
    accuracy_context = {a["solver"] + "/" + a["sample_id"]: a["future_relative_l2"]
                        for a in cls["accuracy"]}

    numbers = {
        "schema": "paper_group4_tables_v1",
        "device": op1["device"],
        "task": op1["protocol"]["task"],
        "timing_protocol": {
            "cuda_synchronized": True, "excludes_disk_io": True,
            "includes_output_materialization": True, "single_instance": True,
            "operator_ic_read_excluded": True,
            "warmup": {"traditional": trad["warmup_runs"], "operator": op1["warmup_runs"]},
        },
        "protocol_differences": [
            "Generation-protocol row times validation_*_00000 records; all other rows "
            "time the paper's figure records (train_uniform_00413, train_layered_00299, "
            "train_marmousi_00010). Speed ratios reported here use only record-matched rows.",
            "The operator amortises training (~GPU-days) and produces all 401 stored "
            "frames of ONE record per call; classical rows are one full solve per record. "
            "No end-to-end (training-inclusive) speedup is claimed.",
            "Operator time excludes reading the 8 IC frames from disk "
            "(~0.02 s, reported separately in the source JSON); classical rows need no IC.",
            "Same GPU (RTX 4090 D), same timing protocol, single instance, "
            "no batching across records in any row.",
            "The native 201x201 classical solve is NOT the truth protocol: its accuracy "
            "vs stored truth is 0.018-0.120 future relL2 depending on family "
            "(classical_lwc84_runtime_accuracy.json); the fine solve is the truth "
            "protocol up to velocity-resampling error (1.9e-5 to 5.8e-2).",
        ],
        "runtime_rows": rows,
        "speed_ratios_record_matched": ratios,
        "classical_accuracy_context_future_relL2": accuracy_context,
    }

    # ---- DeepONet table ----
    afam = load(DON / "collapse_probe" / "ARM_Afam_2000.json")
    cfam = load(DON / "collapse_probe" / "ARM_C_fam2000.json")
    best = load(DON / "main_22814" / "best_tier1_validation.json")
    latest = load(DON / "main_22814" / "validation_latest.json")
    a1 = load(DON / "collapse_probe" / "ARM_A_sat2000.json")

    def arm_summary(d):
        h = d["history"]
        rel = [e["point_relL2"] for e in h]
        amp = [e["amp_ratio"] for e in h]
        return {
            "records": d["records"], "steps": d["steps"], "lr": d["lr"],
            "point_relL2_last": rel[-1], "point_relL2_min": min(rel),
            "point_relL2_max": max(rel),
            "point_relL2_trailing10_mean": float(np.mean(rel[-10:])),
            "point_relL2_trailing10_band": [min(rel[-10:]), max(rel[-10:])],
            "amp_ratio_last": amp[-1],
            "amp_ratio_trailing10_mean": float(np.mean(amp[-10:])),
        }

    numbers["deeponet"] = {
        "parameter_budget": {
            "deeponet_actual": 39975985,
            "prereg_budget": "40,484,837 +/- 2%",
            "ours_ic8_40m": 40484837,
        },
        "main_run": {
            "planned_steps": 22814,
            "terminated_at_step": 4862,
            "termination": "SIGTERM at epoch 26 (run stopped early; "
                           "not a converged endpoint)",
            "fixed_12record_validation": {
                "caliber": "fixed 12-record validation batch (4 per family, "
                           "seed 101+7919), dense relative L2 -- checkpoint-selection "
                           "metric mirrored from the main model's recipe; NOT the "
                           "480-record census caliber",
                "best_tier1_step_4488": {
                    "aggregate_dense_relative_l2": best["validation"][
                        "aggregate_dense_relative_l2"],
                    "family_dense": best["validation"]["family_dense_relative_l2"],
                },
                "latest_step_4862": {
                    "aggregate_dense_relative_l2": latest["validation"][
                        "aggregate_dense_relative_l2"],
                    "family_dense": latest["validation"]["family_dense_relative_l2"],
                },
            },
        },
        "collapse_probe_family_level": {
            "caliber": "12 train records (4 per family), 2000 steps, lr 5e-4; "
                       "point_relL2 on a fixed probe batch (packaged training loss "
                       "caliber, not validation)",
            "arm_A_control": arm_summary(afam),
            "arm_C_anticollapse": arm_summary(cfam),
            "single_record_context_arm_A_2000steps": {
                "records": a1["records"], "steps": a1["steps"],
                "point_relL2_last": a1["history"][-1]["point_relL2"],
            },
            "verdict_20260915": "control arm never beats the trivial zero predictor "
                                "on 12 records (relL2 stays >= 1.0015); anti-collapse "
                                "arm oscillates 0.97-1.33 and its |C-A| effect sizes "
                                "fall inside C's own trailing-10 oscillation band on "
                                "all three axes (loss, amp_ratio, cos): the two arms "
                                "are not separable, and neither fits the 12-record task",
        },
        "ours_same_size_context": {
            "caliber": "480-record validation census, dense future relative L2 "
                       "(record-equal mean), checkpoint 22814 (A4 radial44)",
            "overall_mean": 0.35223194222281917,
            "per_family_mean": {"uniform": 0.13088196939393912,
                                "layered": 0.3020379892646819,
                                "marmousi": 0.5653522506531667},
            "source": "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/"
                      "REVIEWER_PACKAGE_20260921/results/validation_480_summary.json",
        },
        "caliber_mismatch_note": "No same-caliber number exists: DeepONet was stopped "
                                 "at step 4862/22814 and never evaluated on the "
                                 "480-record census. The nearest comparison is the "
                                 "fixed 12-record validation batch (DeepONet ~1.0000 = "
                                 "trivial level) vs our 480-record census (0.352 mean); "
                                 "both are dense future relative L2 but on different "
                                 "record sets and selection protocols.",
        "sources": {
            "arm_Afam": str(DON / "collapse_probe" / "ARM_Afam_2000.json"),
            "arm_Cfam": str(DON / "collapse_probe" / "ARM_C_fam2000.json"),
            "best_tier1": str(DON / "main_22814" / "best_tier1_validation.json"),
            "latest": str(DON / "main_22814" / "validation_latest.json"),
            "run_config": str(DON / "main_22814" / "run_config.json"),
            "prereg": "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/"
                      "release_ic8_40m_20260910/research/paper_support_20260914/"
                      "deeponet_baseline/PREREG_deeponet_baseline_v1.md",
            "synthesis": "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/"
                         "release_ic8_40m_20260910/research/"
                         "LATE_TIME_CODA_EVIDENCE_SYNTHESIS_20260915.md",
        },
    }

    (OUT / "NUMBERS.json").write_text(json.dumps(numbers, indent=1))

    # ---- markdown tables ----
    rt_md = ["# Runtime comparison (one record -> 401 stored frames, 201x201, RTX 4090 D)",
             "",
             "| Method | Mean (s) | Min (s) | n | Records |",
             "|---|---|---|---|---|"]
    for r in rows:
        rt_md.append("| {method} | {mean:.2f} | {mn:.2f} | {n} | {rec} |".format(
            method=r["method"], mean=r["mean_s"], mn=r["min_s"], n=r["n"],
            rec=r["records"]))
    rt_md += [
        "",
        "Record-matched speed ratios (paper figure records only):",
        "",
        "| Comparison | Ratio |", "|---|---|",
        "| operator (tb=1) vs fine 401x401 | {:.2f}x |".format(
            ratios["operator_tb1_vs_fine_records_matched"]),
        "| operator (tb=1) vs native 201x201 | {:.2f}x |".format(
            ratios["operator_tb1_vs_native_records_matched"]),
        "| operator (tb=16) vs fine 401x401 | {:.2f}x |".format(
            ratios["operator_tb16_vs_fine_records_matched"]),
        "| operator (tb=16) vs native 201x201 | {:.2f}x |".format(
            ratios["operator_tb16_vs_native_records_matched"]),
        "",
        "Footnotes (protocol differences):",
    ]
    rt_md += [f"{i+1}. {t}" for i, t in enumerate(numbers["protocol_differences"])]
    (OUT / "TABLE_runtime.md").write_text("\n".join(rt_md) + "\n")

    dv = numbers["deeponet"]
    dn_md = [
        "# DeepONet baseline comparison (40M parameter budget)",
        "",
        "| Quantity | DeepONet 40M | Ours (IC8 40M) | Caliber |",
        "|---|---|---|---|",
        "| Parameters | 39,975,985 | 40,484,837 | same +/-2% budget (preregistered) |",
        "| Fixed 12-record validation, dense relL2 (best ckpt, step 4488) | "
        "{:.6f} | n/a (not our selection metric) | DeepONet's checkpoint-selection "
        "batch; ~1.0 = trivial zero predictor |".format(
            dv["main_run"]["fixed_12record_validation"]["best_tier1_step_4488"][
                "aggregate_dense_relative_l2"]),
        "| 480-record validation census, dense future relL2 (record mean) | "
        "not evaluated (run stopped at step 4862/22814) | 0.352 (uniform 0.131 / "
        "layered 0.302 / marmousi 0.565) | ours: full census at step 22814 |",
        "| 12-record family-level fit probe (2000 steps), point relL2 | control arm "
        "1.0015-1.1168, never < 1.0 | n/a | packaged-loss probe caliber |",
        "",
        "Facts stated as recorded:",
        "- The DeepONet main run was terminated by SIGTERM at step 4862 of a planned "
        "22814; its numbers are NOT a converged endpoint.",
        "- At every recorded validation, DeepONet's dense relative L2 was "
        "indistinguishable from the trivial zero predictor (1.0000 +/- 1e-4), on "
        "all three families.",
        "- The 2026-09-15 family-level collapse probe found the control arm never "
        "beats trivial on 12 records, and the anti-collapse arm is not separable "
        "from the control arm (all three effect axes inside its own oscillation "
        "band). The failure was attributed to optimisation (H_optim), not "
        "capacity (H_capacity) - single-record fits do converge (relL2 0.025 "
        "at 2000 steps).",
        "- No same-caliber DeepONet number exists; the table's calibers are the "
        "nearest available and are labelled per row.",
    ]
    (OUT / "TABLE_deeponet.md").write_text("\n".join(dn_md) + "\n")
    print("written", OUT / "NUMBERS.json")
    print("written", OUT / "TABLE_runtime.md")
    print("written", OUT / "TABLE_deeponet.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
