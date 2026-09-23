#!/usr/bin/env python3
"""Self-check for the reviewer package.

Three independent checks, each printing PASS or FAIL:

  1. code provenance  -- every file in code/ matches the sha256 recorded by the
                         run that produced the reported results (PROVENANCE.tsv)
  2. band identity    -- the per-band decomposition re-derives the independently
                         measured pooled error: total^2 = sum E_share*relL2^2
  3. ceiling algebra  -- the quoted improvement ceilings follow from that table

Check 2 and 3 need only the standard library.  Check 1 needs nothing but hashlib.
Run from the package root:  python3 verify_package.py
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
TOL = 1e-5


def check_provenance() -> bool:
    manifest = ROOT / "code" / "PROVENANCE.tsv"
    if not manifest.exists():
        print("FAIL  provenance: PROVENANCE.tsv missing")
        return False
    rows = [l.split("\t") for l in manifest.read_text().splitlines()[1:] if l.strip()]
    bad = []
    for status, rel, recorded, _ in rows:
        f = ROOT / "code" / rel
        if not f.exists():
            bad.append(f"{rel}: absent")
            continue
        actual = hashlib.sha256(f.read_bytes()).hexdigest()
        if actual != recorded:
            bad.append(f"{rel}: {actual[:12]} != recorded {recorded[:12]}")
    if bad:
        print(f"FAIL  provenance: {len(bad)} of {len(rows)} files do not match")
        for b in bad[:8]:
            print("        ", b)
        return False
    print(f"PASS  provenance: {len(rows)} files match the sha256 recorded by the producing run")
    return True


def _table():
    p = ROOT / "results" / "per_record_aggregates" / "error_spectrum_by_band.json"
    return json.loads(p.read_text())


def check_band_identity() -> bool:
    data = _table()
    groups = dict(data["families"])
    groups["all (pooled)"] = data["pooled_all_families"]
    ok = True
    for name, g in groups.items():
        shares = sum(b["E_share"] for b in g["bands"])
        derived = math.sqrt(sum(b["E_share"] * b["relative_l2"] ** 2 for b in g["bands"]))
        measured = g["pooled_future_relative_l2"]
        good = abs(derived - measured) < TOL and abs(shares - 1.0) < TOL
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  identity {name:14s} "
              f"measured {measured:.5f}  from bands {derived:.5f}  E_shares sum {shares:.5f}")
    return ok


def check_ceilings() -> bool:
    data = _table()
    groups = dict(data["families"])
    groups["all (pooled)"] = data["pooled_all_families"]
    ok = True
    for name, g in groups.items():
        bands = g["bands"]
        best = bands[0]["relative_l2"]
        tail = math.sqrt(sum(b["E_share"] * (0.0 if i >= 2 else b["relative_l2"] ** 2)
                             for i, b in enumerate(bands)))
        b2 = math.sqrt(sum(b["E_share"] * ((best if i == 1 else b["relative_l2"]) ** 2)
                           for i, b in enumerate(bands)))
        good = (abs(tail - g["ceiling_tail_bands_zeroed"]) < TOL
                and abs(b2 - g["ceiling_band2_to_band1"]) < TOL
                and abs(best - g["ceiling_fully_flattened"]) < TOL)
        ok &= good
        base = g["pooled_future_relative_l2"]
        print(f"{'PASS' if good else 'FAIL'}  ceilings {name:14s} "
              f"tail→0 {tail:.4f} ({100*(1-tail/base):.1f}%)  "
              f"flattened {best:.4f} ({100*(1-best/base):.1f}%)  "
              f"gap to 0.1 {best/0.1:.2f}x")
    return ok


def main() -> int:
    print("Reviewer package self-check\n" + "-" * 78)
    results = [check_provenance()]
    print()
    results.append(check_band_identity())
    print()
    results.append(check_ceilings())
    print("-" * 78)
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
