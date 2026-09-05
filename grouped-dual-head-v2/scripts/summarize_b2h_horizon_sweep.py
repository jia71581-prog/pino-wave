#!/usr/bin/env python3
"""Poll / summarize the B2-H training-horizon diagnostic.

--poll  : print per-run training progress (epochs done, latest in-window agg).
--report: build the strict-rollout curve (agg + late bin vs training horizon),
          write a PNG and a markdown table.  Physical baseline is horizon-
          independent and drawn as a reference line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "b2h_horizon_sweep"
RUN_ROOT = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1"
    "/pretraining/b2h/horizon_sweep"
)
HORIZONS = (8, 16, 32, 64)


def _last_metric(horizon: int) -> dict | None:
    path = RUN_ROOT / f"h{horizon:02d}" / "run" / "metrics.jsonl"
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def poll() -> None:
    for horizon in HORIZONS:
        row = _last_metric(horizon)
        if row is None:
            print(f"h{horizon:02d}: (no metrics yet)")
            continue
        m = row["metrics"]
        b = row["physical_baseline_metrics"]
        print(
            f"h{horizon:02d}: ep{row['epoch']:>2} "
            f"in_window_agg={m['aggregate_relative_l2']:.4f} "
            f"base={b['aggregate_relative_l2']:.4f} "
            f"gate={row['gate_effective']:.3f} "
            f"rscale={row['residual_scale']:.2e} "
            f"peak={row['peak_cuda_gib_rank0']:.2f}G "
            f"t={row['elapsed_seconds']:.0f}s"
        )


def _strict(horizon: int, which: str) -> dict | None:
    path = RES / f"h{horizon:02d}_{which}_strict.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def report() -> None:
    rows = []
    baseline_agg = None
    for horizon in HORIZONS:
        rec = _strict(horizon, "latest")
        if rec is None:
            print(f"h{horizon:02d}: strict json missing, skipping")
            continue
        learned = rec["learned"]
        base = rec["physical_baseline"]
        baseline_agg = base["aggregate_relative_l2"]
        rows.append(
            {
                "horizon": horizon,
                "epoch": rec["checkpoint_epoch"],
                "learned_agg": learned["aggregate_relative_l2"],
                "learned_late": learned["time_bin_relative_l2"]["late"],
                "learned_mid": learned["time_bin_relative_l2"]["middle"],
                "base_agg": base["aggregate_relative_l2"],
                "base_late": base["time_bin_relative_l2"]["late"],
                "phase": learned["phase_correlation"],
            }
        )

    md = ["# B2-H training-horizon diagnostic (strict 401-step free rollout)", ""]
    md.append(f"Physical baseline (gate=0, horizon-independent): agg = {baseline_agg:.4f}")
    md.append("")
    md.append("| train horizon | ckpt ep | learned agg | vs base | learned late | learned mid | phase corr |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        delta = r["learned_agg"] - r["base_agg"]
        sign = "+" if delta >= 0 else ""
        md.append(
            f"| {r['horizon']} | {r['epoch']} | {r['learned_agg']:.4f} | "
            f"{sign}{delta:.4f} | {r['learned_late']:.4f} | "
            f"{r['learned_mid']:.4f} | {r['phase']:.3f} |"
        )
    md_text = "\n".join(md) + "\n"
    (RES / "SUMMARY.md").write_text(md_text)
    print(md_text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped: {exc}]")
        return
    if not rows:
        return
    ink = "#1b1b1b"
    xs = [r["horizon"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    ax.plot(xs, [r["learned_agg"] for r in rows], "-o", color="#c0392b",
            label="learned residual (strict agg)")
    ax.plot(xs, [r["learned_late"] for r in rows], "--s", color="#e08e0b",
            label="learned residual (late bin)")
    ax.axhline(rows[0]["base_agg"], color="#2c6fbb", lw=1.6,
               label=f"physical baseline agg = {rows[0]['base_agg']:.3f}")
    ax.axhline(rows[0]["base_late"], color="#2c6fbb", lw=1.0, ls=":",
               label=f"physical baseline late = {rows[0]['base_late']:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("training rollout horizon (saved steps)", color=ink)
    ax.set_ylabel("strict 401-step free-rollout rel. L2", color=ink)
    ax.set_title("B2-H: does a longer training horizon tame free rollout?", color=ink)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = RES / "b2h_horizon_curve.png"
    fig.savefig(out)
    print(f"[wrote {out}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.poll:
        poll()
    if args.report:
        report()
    if not (args.poll or args.report):
        poll()


if __name__ == "__main__":
    main()
