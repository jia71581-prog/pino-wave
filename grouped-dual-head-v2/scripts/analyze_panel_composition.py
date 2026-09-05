#!/usr/bin/env python
"""Verify the record-weighted composition and reachability bound of a validation panel.

Reproducibility tool for the acoustic-operator research honesty requirement.
Given a run's ``metrics.jsonl`` and an epoch, it:
  1. reads the exact per-record ``source_relative_l2`` (== the 48-record vector that
     ``aggregate_relative_l2 = np.mean(...)`` is computed over, streaming_metrics.py:227);
  2. reports ``record_count``, the medium-GROUP family counts (from ``medium_relative_l2``
     keys, which may be fewer than records when a medium group holds >1 record), and the
     reported per-family means;
  3. integer-solves the per-family RECORD split (n_layered, n_marmousi, n_uniform) that best
     reconstructs ``aggregate = sum_f n_f/N * family_mean_f`` -- this is the record-weighting
     that actually gates the aggregate (the source keys do NOT encode family, so the split is
     recovered by search, cross-checked against the group counts);
  4. prints the reachability arithmetic for the agg<0.05 hard goal under that composition.

This exists because the split was mis-guessed once (u=12/l=26/m=10, "layered 54%") before being
verified as l=19/m=19/u=10 (~40/40/21). Run it instead of eyeballing; it emits raw numbers only.

Usage:
  python scripts/analyze_panel_composition.py --metrics <run>/metrics.jsonl --epoch 12
  python scripts/analyze_panel_composition.py --metrics <run>/metrics.jsonl --epoch 12 --target 0.05
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json


def _load_epoch_row(path: str, epoch: int | None) -> dict:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    if epoch is None:
        row = rows[-1]
    else:
        matches = [r for r in rows if r.get("epoch") == epoch]
        if not matches:
            have = sorted({r.get("epoch") for r in rows})
            raise SystemExit(f"epoch {epoch} not found in {path}; have epochs {have}")
        row = matches[-1]
    return row


def _family_of(key: str) -> str:
    parts = key.split(":")
    return parts[1] if len(parts) >= 2 else "?"


def solve_record_split(family_means: dict[str, float], agg: float, n: int):
    """Brute-force the integer per-family record counts best reconstructing the aggregate."""
    fams = sorted(family_means)
    if len(fams) != 3:
        return None
    target = agg * n
    best = None
    fa, fb, fc = fams
    for na in range(0, n + 1):
        for nb in range(0, n - na + 1):
            nc = n - na - nb
            val = family_means[fa] * na + family_means[fb] * nb + family_means[fc] * nc
            err = abs(val - target)
            if best is None or err < best[0]:
                best = (err, {fa: na, fb: nb, fc: nc}, val / n)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="run metrics.jsonl to analyze")
    ap.add_argument("--epoch", type=int, default=None, help="epoch to analyze (default: last row)")
    ap.add_argument("--target", type=float, default=0.05, help="hard aggregate goal (default 0.05)")
    ap.add_argument("--family-target", type=float, default=0.08, help="companion family goal (default 0.08)")
    args = ap.parse_args(argv)

    row = _load_epoch_row(args.metrics, args.epoch)
    m = row.get("metrics", row)
    agg = m.get("aggregate_relative_l2")
    fam_means = m.get("family_relative_l2", {})
    src = m.get("source_relative_l2", {})
    med = m.get("medium_relative_l2", {})
    tb = m.get("time_bin_relative_l2", {})
    rc = m.get("record_count")

    print(f"metrics={args.metrics}")
    print(f"epoch={row.get('epoch')}  record_count={rc}  "
          f"n(source_relative_l2)={len(src)}  n(medium groups)={len(med)}")
    print(f"aggregate_relative_l2 = {agg}")
    if src:
        import statistics
        recon = statistics.fmean(src.values())
        print(f"  cross-check np.mean(source_relative_l2) = {recon:.6f} "
              f"(should equal aggregate; delta {abs(recon - agg):.2e})")
    print(f"phase_correlation = {m.get('phase_correlation')}")
    print(f"family_relative_l2 = {{ {', '.join(f'{k}: {v:.5f}' for k, v in sorted(fam_means.items()))} }}")
    if tb:
        print(f"time_bin_relative_l2 = {{ {', '.join(f'{k}: {v:.4f}' for k, v in tb.items())} }}")

    gc = collections.Counter(_family_of(k) for k in med)
    print(f"\nmedium-GROUP family counts (may be < records): {dict(gc)}")

    N = rc if isinstance(rc, int) and rc > 0 else len(src)
    if fam_means and agg is not None and N:
        best = solve_record_split(fam_means, agg, N)
        if best is not None:
            err, split, mean = best
            total = sum(split.values()) or 1
            weights = ", ".join(f"{k} {v} ({100 * v / total:.1f}%)" for k, v in sorted(split.items()))
            print(f"\nbest integer RECORD split over N={N}: {weights}")
            print(f"  reconstructed record-weighted agg = {mean:.6f}  (|err| {err:.2e})")

            print(f"\nREACHABILITY for aggregate < {args.target} (record-weighted):")
            allft = args.family_target
            print(f"  all families at {allft} family target      -> agg = {allft:.4f}  "
                  f"{'PASS' if allft < args.target else 'FAIL'}")
            # scattering families (non-uniform) to target, uniform held at current
            uni = fam_means.get("uniform")
            if uni is not None:
                scat = {k: (args.target if k != "uniform" else uni) for k in split}
                v = sum(scat[k] * split[k] for k in split) / total
                print(f"  non-uniform families to {args.target}, uniform {uni:.4f} -> agg = {v:.4f}  "
                      f"{'PASS' if v < args.target else 'FAIL'}")
            v = args.target
            print(f"  every family to {args.target}                 -> agg = {v:.4f}  "
                  f"{'PASS' if v < args.target else 'FAIL (boundary)'}")
            print("  => needed per-family cuts to hit agg<{:.2f}:".format(args.target))
            for k in sorted(fam_means):
                cur = fam_means[k]
                cut = 100 * (1 - args.target / cur) if cur else 0.0
                print(f"       {k:9s} {cur:.4f} -> ~{args.target:.2f}  (~{cut:.0f}% error cut)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
