#!/usr/bin/env python
"""Mechanical, scope-locked switch-gate evaluator for the saved-time arch ladder.

The whole research loop hinges on the pre-registered ep4/8/12 STRICT-panel switch
gate.  Judging it by eye invites two integrity failures the mandate forbids: (1)
mixing validation scopes (the light ``fixed_panel`` agg is ~0.01 lower than the
strict ``fixed_full_time_panel`` agg and is NOT comparable), and (2) declaring a
win on an aggregate-only drop that is really smoothing.  This script removes the
human from that judgement: it reads a run's ``metrics.jsonl``, extracts ONLY the
strict ``fixed_full_time_panel`` rows, and applies the frozen gate arithmetic.

Sacred scope-locked baselines (fixed_full_time_panel == STRICT; provenance below):
  * opt16   strict  ep28: aggregate=0.2529994053788361, phase=0.7847295582497394
            (user-pinned panel-mixing prohibition baseline).
  * warp_r1 strict: read live from its own metrics.jsonl (best-aggregate row) so
            the reference tracks the actual parent ceiling, not a transcribed const.

Switch gate (a structural corrector must COMPLETE transport, not just smooth):
    aggregate < 0.240  AND  late tbin < 0.30  AND  layered family < 0.27
    AND phase_correlation >= 0.80
An aggregate-only drop (agg down while any of late/layered/phase fail) => the
corrector smooths but does not transport-complete => escalate to the next rung.

This is only the switch gate; the unchanged HARD goal remains aggregate < 0.05 and
every family < 0.08 on the same fixed protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

STRICT_SCOPE = "fixed_full_time_panel"

# opt16 strict ep28 -- user-pinned, never recomputed (the run is archived).
OPT16_STRICT = {"aggregate": 0.2529994053788361, "phase": 0.7847295582497394}

# Pre-registered switch-gate thresholds (frozen).
GATE = {
    "aggregate_below": 0.240,
    "late_below": 0.30,
    "layered_below": 0.27,
    "phase_at_least": 0.80,
}


def strict_rows(metrics_path: str | Path) -> dict[int, dict]:
    """epoch -> metrics dict, for STRICT fixed_full_time_panel rows only."""
    rows: dict[int, dict] = {}
    path = Path(metrics_path)
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "epoch":
            continue
        if record.get("validation_scope") != STRICT_SCOPE:
            continue
        rows[int(record["epoch"])] = record["metrics"]
    return rows


def _agg(metrics: Mapping) -> float:
    if "aggregate_relative_l2" in metrics:
        return float(metrics["aggregate_relative_l2"])
    return float(metrics["aggregate_floored_relative_l2"])


def best_strict(metrics_path: str | Path) -> tuple[int, dict] | None:
    """Return (epoch, metrics) of the min-aggregate strict row, or None."""
    rows = strict_rows(metrics_path)
    if not rows:
        return None
    epoch = min(rows, key=lambda e: _agg(rows[e]))
    return epoch, rows[epoch]


def evaluate_gate(metrics: Mapping) -> dict:
    """Apply the frozen gate to one strict-panel metrics dict."""
    agg = _agg(metrics)
    late = float(metrics["time_bin_relative_l2"]["late"])
    layered = float(metrics["family_relative_l2"]["layered"])
    phase = float(metrics["phase_correlation"])
    checks = {
        "aggregate": (agg, agg < GATE["aggregate_below"], f"< {GATE['aggregate_below']}"),
        "late_tbin": (late, late < GATE["late_below"], f"< {GATE['late_below']}"),
        "layered_family": (layered, layered < GATE["layered_below"], f"< {GATE['layered_below']}"),
        "phase_correlation": (phase, phase >= GATE["phase_at_least"], f">= {GATE['phase_at_least']}"),
    }
    passed = all(ok for _, ok, _ in checks.values())
    # smoothing signature: aggregate improved but transport metrics did not clear
    agg_only = checks["aggregate"][1] and not (
        checks["late_tbin"][1] and checks["layered_family"][1] and checks["phase_correlation"][1]
    )
    return {"passed": passed, "aggregate_only_smoothing": agg_only, "checks": checks}


def decompose(metrics: Mapping) -> dict | None:
    """Attribute error to the coarse field vs the correction stack.

    Returns per-region (coarse -> final) values and the fractional improvement the
    correction stack buys, or None if the strict row carries no coarse_metrics.
    A small improvement in a region means the correction stack is CAPPED there --
    i.e. that region's residual is where the next structural rung must act.
    """
    coarse = metrics.get("coarse_metrics")
    if not coarse:
        return None

    def _pair(final_v, coarse_v):
        final_v, coarse_v = float(final_v), float(coarse_v)
        frac = (coarse_v - final_v) / coarse_v if coarse_v > 0 else 0.0
        return {"coarse": coarse_v, "final": final_v, "improvement_frac": frac}

    out = {"aggregate": _pair(_agg(metrics), _agg(coarse))}
    cf, ff = coarse.get("family_relative_l2", {}), metrics.get("family_relative_l2", {})
    for fam in sorted(ff):
        if fam in cf:
            out[f"family:{fam}"] = _pair(ff[fam], cf[fam])
    ct, ft = coarse.get("time_bin_relative_l2", {}), metrics.get("time_bin_relative_l2", {})
    for tb in sorted(ft):
        if tb in ct:
            out[f"tbin:{tb}"] = _pair(ft[tb], ct[tb])
    return out


def _fmt_decompose(dec: dict) -> str:
    lines = ["-- coarse -> correction attribution (small improvement = correction capped here) --"]
    for name, d in dec.items():
        lines.append(
            f"  {name:16s} coarse={d['coarse']:.4f} -> final={d['final']:.4f}  "
            f"({d['improvement_frac']*100:5.1f}% improvement)"
        )
    return "\n".join(lines)


def _fmt(verdict: dict) -> str:
    lines = []
    for name, (value, ok, thresh) in verdict["checks"].items():
        mark = "PASS" if ok else "FAIL"
        lines.append(f"  [{mark}] {name:18s} = {value:.6f}   (need {thresh})")
    tail = "GATE PASS -> transport-completing" if verdict["passed"] else (
        "GATE FAIL: AGGREGATE-ONLY SMOOTHING -> escalate next rung"
        if verdict["aggregate_only_smoothing"]
        else "GATE FAIL"
    )
    return "\n".join(lines) + "\n  => " + tail


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="run metrics.jsonl to judge")
    parser.add_argument("--epoch", type=int, default=None,
                        help="strict epoch to judge (default: best-aggregate strict row)")
    parser.add_argument("--reference-metrics", default=None,
                        help="parent run metrics.jsonl for a scope-correct ceiling row")
    parser.add_argument("--decompose", action="store_true",
                        help="also print coarse->correction error attribution")
    args = parser.parse_args(argv)

    rows = strict_rows(args.metrics)
    if not rows:
        print(f"NO strict ({STRICT_SCOPE}) rows yet in {args.metrics}")
        return 2
    if args.epoch is not None:
        if args.epoch not in rows:
            print(f"epoch {args.epoch} has no strict row; available: {sorted(rows)}")
            return 2
        epoch, metrics = args.epoch, rows[args.epoch]
    else:
        epoch = min(rows, key=lambda e: _agg(rows[e]))
        metrics = rows[epoch]

    print(f"== switch-gate verdict: {args.metrics}  strict epoch {epoch} ==")
    verdict = evaluate_gate(metrics)
    print(_fmt(verdict))

    if args.decompose:
        dec = decompose(metrics)
        print("\n" + (_fmt_decompose(dec) if dec else "  (no coarse_metrics in this strict row)"))

    print(f"\n-- scope-locked baselines ({STRICT_SCOPE}) --")
    print(f"  opt16 strict ep28: aggregate={OPT16_STRICT['aggregate']:.6f} "
          f"phase={OPT16_STRICT['phase']:.6f}")
    if args.reference_metrics:
        ref = best_strict(args.reference_metrics)
        if ref:
            r_ep, r_m = ref
            print(f"  parent strict best (ep{r_ep}): aggregate={_agg(r_m):.6f} "
                  f"late={float(r_m['time_bin_relative_l2']['late']):.6f} "
                  f"layered={float(r_m['family_relative_l2']['layered']):.6f} "
                  f"phase={float(r_m['phase_correlation']):.6f}")
    print(f"\n  HARD goal (unchanged): aggregate < 0.05 AND every family < 0.08")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
