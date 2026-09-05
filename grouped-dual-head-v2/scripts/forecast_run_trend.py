"""Scope-separated quantitative trend forecast for a live/finished run (CODEX 2026-07-30
'trend forecasting' addendum, deliverable #2).

Reads a run's metrics.jsonl and, PARTITIONED BY validation_scope (never mixing panels),
forecasts aggregate + every family + every time bin with:
  * raw trajectory;
  * robust Theil-Sen slope + 95% CI (last-window);
  * exponential-asymptote fit m(e)=a+b*exp(-k*e) with an IDENTIFIABILITY gate
    (relative std-err of the asymptote a, and fit residual);
  * target gap and a target-crossing ETA ONLY when the fit is identifiable AND the
    asymptote lies below the target. Otherwise reports "not identifiable / not reachable".

HONESTY: never extrapolates a noisy ~0.23 curve to 0.05 just because a linear slope is
negative. If the asymptote is not statistically pinned, it says so.

Run (CPU): CUDA_VISIBLE_DEVICES="" python scripts/forecast_run_trend.py \
    --metrics <run>/metrics.jsonl --out <run>/trend_forecast.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import optimize, stats

AGG_TARGET = 0.05
FAMILY_TARGET = 0.08
LAST_WINDOW = 8          # epochs for the robust local slope
MIN_POINTS = 6           # need this many epochs before any fit is attempted


def _load(path):
    rows = [json.loads(l) for l in open(path)]
    by_scope = {}
    for r in rows:
        scope = r.get("validation_scope", "unknown")
        m = r.get("metrics")
        if not isinstance(m, dict) or "aggregate_relative_l2" not in m:
            continue
        by_scope.setdefault(scope, []).append((int(r["epoch"]), m))
    for s in by_scope:
        by_scope[s].sort(key=lambda t: t[0])
    return by_scope


def _series(rows, extract):
    e, y = [], []
    for ep, m in rows:
        v = extract(m)
        if v is not None and np.isfinite(v):
            e.append(ep); y.append(float(v))
    return np.asarray(e, float), np.asarray(y, float)


def _theil(e, y):
    if len(e) < 3:
        return None
    w = min(LAST_WINDOW, len(e))
    ew, yw = e[-w:], y[-w:]
    res = stats.theilslopes(yw, ew, 0.95)
    return {"slope": float(res.slope), "slope_lo": float(res.low_slope),
            "slope_hi": float(res.high_slope), "window": int(w),
            "intercept": float(res.intercept)}


def _exp_fit(e, y, target):
    """m(e)=a+b*exp(-k*e). Returns fit + identifiability verdict + ETA (if valid)."""
    if len(e) < MIN_POINTS:
        return {"identifiable": False, "reason": f"<{MIN_POINTS} epochs"}
    e0 = e - e[0]
    a0 = float(min(y.min(), y[-1]))
    b0 = float(y[0] - a0) or 0.1
    try:
        popt, pcov = optimize.curve_fit(
            lambda x, a, b, k: a + b * np.exp(-k * x), e0, y,
            p0=[a0, b0, 0.1], bounds=([-1.0, -5.0, 1e-4], [5.0, 5.0, 5.0]),
            maxfev=20000)
    except Exception as ex:  # noqa: BLE001
        return {"identifiable": False, "reason": f"fit failed: {type(ex).__name__}"}
    a, b, k = popt
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    pred = a + b * np.exp(-k * e0)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    # identifiability: asymptote a must be pinned (rel std-err small) and residual small
    a_relerr = float(perr[0] / (abs(a) + 1e-9))
    identifiable = (a_relerr < 0.5) and (rmse < 0.05) and (perr[0] < 0.1)
    out = {"identifiable": bool(identifiable), "asymptote_a": float(a),
           "a_stderr": float(perr[0]), "a_relerr": a_relerr, "b": float(b),
           "k": float(k), "fit_rmse": rmse}
    if not identifiable:
        out["reason"] = ("asymptote not pinned (a_relerr>=0.5 or stderr>=0.1 or rmse>=0.05); "
                         "cannot reliably extrapolate")
        return out
    # ETA only if asymptote is below target (else target unreachable by this trend)
    if a >= target:
        out["target_reachable"] = False
        out["reason"] = f"identifiable asymptote a={a:.4f} >= target {target} -> NOT reachable"
        return out
    if b <= 0:  # increasing toward asymptote from below -> already below? guard
        out["target_reachable"] = True if y[-1] < target else False
        out["eta_epoch"] = None
        return out
    # solve a + b exp(-k (e-e0)) = target -> e = e0 - ln((target-a)/b)/k
    frac = (target - a) / b
    if frac <= 0:
        out["target_reachable"] = False
        out["reason"] = "target below asymptote branch (unreachable)"
        return out
    eta = e[0] + (-np.log(frac) / k)
    out["target_reachable"] = True
    out["eta_epoch"] = float(eta)
    out["eta_epochs_from_now"] = float(eta - e[-1])
    return out


def _forecast_one(name, e, y, target):
    rep = {"name": name, "n_epochs": int(len(e)),
           "epochs": e.astype(int).tolist(), "values": [round(v, 5) for v in y]}
    if len(y):
        rep["latest"] = round(float(y[-1]), 5)
        rep["best"] = round(float(y.min()), 5)
        rep["target"] = target
        rep["gap_to_target"] = round(float(y[-1] - target), 5)
    rep["theil_sen"] = _theil(e, y)
    rep["exp_asymptote"] = _exp_fit(e, y, target)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    by_scope = _load(args.metrics)
    report = {"metrics_path": args.metrics, "agg_target": AGG_TARGET,
              "family_target": FAMILY_TARGET, "scopes": {}}
    for scope, rows in by_scope.items():
        sc = {"n_epochs": len(rows), "forecasts": {}}
        e, y = _series(rows, lambda m: m.get("aggregate_relative_l2"))
        sc["forecasts"]["aggregate"] = _forecast_one("aggregate", e, y, AGG_TARGET)
        for fam in sorted(rows[-1][1].get("family_relative_l2", {})):
            e, y = _series(rows, lambda m, f=fam: m.get("family_relative_l2", {}).get(f))
            sc["forecasts"][f"family:{fam}"] = _forecast_one(f"family:{fam}", e, y, FAMILY_TARGET)
        for tb in sorted(rows[-1][1].get("time_bin_relative_l2", {})):
            e, y = _series(rows, lambda m, t=tb: m.get("time_bin_relative_l2", {}).get(t))
            sc["forecasts"][f"time_bin:{tb}"] = _forecast_one(f"time_bin:{tb}", e, y, AGG_TARGET)
        report["scopes"][scope] = sc

    # human-readable console summary
    for scope, sc in report["scopes"].items():
        print(f"\n===== scope={scope}  ({sc['n_epochs']} epochs) =====")
        for key, fc in sc["forecasts"].items():
            ts = fc.get("theil_sen") or {}
            ex = fc.get("exp_asymptote") or {}
            slope = ts.get("slope")
            slope_s = (f"slope={slope:+.5f}/ep [{ts.get('slope_lo'):+.5f},{ts.get('slope_hi'):+.5f}]"
                       if slope is not None else "slope=n/a")
            if ex.get("identifiable"):
                if ex.get("target_reachable") and ex.get("eta_epoch") is not None:
                    verdict = f"asymptote={ex['asymptote_a']:.4f} REACHES {fc['target']} ~ep{ex['eta_epoch']:.0f}"
                else:
                    verdict = f"asymptote={ex['asymptote_a']:.4f} (>= target {fc['target']}) NOT reachable"
            else:
                verdict = f"NOT identifiable ({ex.get('reason','')})"
            print(f"  {key:22s} latest={fc.get('latest')} best={fc.get('best')} "
                  f"gap={fc.get('gap_to_target')}  {slope_s}")
            print(f"  {'':22s} -> {verdict}")

    out = args.out or (args.metrics.rsplit("/", 1)[0] + "/trend_forecast.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
