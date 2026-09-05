#!/usr/bin/env python3
"""Human-readable live monitor for a full-support training log.

Tails the JSON training log and prints:
  * every optimizer step: loss (total + frame/spectrum) with a stability marker
    (down/up vs the previous step and a short moving trend);
  * every epoch: the validation relative L2 (agg + per-family + late bin).

Reads the log the trainer already writes (each optimizer_update / epoch event),
so it needs no change to the training code and does not disturb the run.

Usage: python scripts/monitor_training.py results/tempop_full.log [--from-start]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _fmt_step(d: dict, prev_total: float | None) -> str:
    lc = d.get("loss_components", {})
    total = lc.get("total", float("nan"))
    frame = lc.get("frame", float("nan"))
    spec = lc.get("spectrum", float("nan"))
    ep = d.get("epoch", "?")
    up = d.get("update", "?")
    per = d.get("updates_per_epoch", "?")
    if prev_total is None:
        marker = "  ·"
    elif total < prev_total - 1e-6:
        marker = " ↓"          # decreasing (good)
    elif total > prev_total + 1e-6:
        marker = " ↑"          # increased this step
    else:
        marker = "  ="
    return (f"e{ep} step {up}/{per}  loss={total:.4f}{marker}"
            f"  (frame {frame:.4f} | spectrum {spec:.4f})")


def _fmt_epoch(d: dict) -> str:
    m = d.get("metrics", {})
    agg = m.get("aggregate_relative_l2", float("nan"))
    fam = m.get("family_relative_l2", {})
    tb = m.get("time_bin_relative_l2", {})
    pc = m.get("phase_correlation", float("nan"))
    scope = d.get("validation_scope", "?")
    fam_s = " ".join(f"{k}={v:.3f}" for k, v in fam.items())
    return (f"\n===== EPOCH {d.get('epoch')} VALIDATION [{scope}] =====\n"
            f"  relative L2 (agg) = {agg:.4f}   phase_corr = {pc:.3f}\n"
            f"  by family: {fam_s}\n"
            f"  by time:   early={tb.get('early', -1):.3f} middle={tb.get('middle', -1):.3f} "
            f"late={tb.get('late', -1):.3f}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--from-start", action="store_true", help="replay existing lines first")
    ap.add_argument("--poll", type=float, default=5.0)
    args = ap.parse_args(argv)
    path = Path(args.log)

    prev_total: float | None = None
    best_agg: float | None = None
    pos = 0
    if not args.from_start:
        # skip to end so we only show new activity
        while not path.exists():
            time.sleep(args.poll)
        pos = path.stat().st_size

    while True:
        if not path.exists():
            time.sleep(args.poll)
            continue
        with path.open() as handle:
            handle.seek(pos)
            for line in handle:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = d.get("event")
                if event == "optimizer_update":
                    print(_fmt_step(d, prev_total), flush=True)
                    prev_total = d.get("loss_components", {}).get("total", prev_total)
                elif event == "epoch":
                    print(_fmt_epoch(d), flush=True)
                    agg = d.get("metrics", {}).get("aggregate_relative_l2")
                    if agg is not None and (best_agg is None or agg < best_agg):
                        best_agg = agg
                        print(f"  >>> NEW BEST agg = {best_agg:.4f}\n", flush=True)
            pos = handle.tell()
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
