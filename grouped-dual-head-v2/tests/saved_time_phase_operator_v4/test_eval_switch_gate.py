"""Tests for the mechanical switch-gate evaluator (scripts/eval_switch_gate.py).

Pin the two integrity properties the gate exists to enforce:
  * it reads ONLY strict fixed_full_time_panel rows (never the light panel), and
  * an aggregate-only drop that leaves late/layered/phase failing is flagged as
    smoothing, NOT a pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_switch_gate import evaluate_gate, strict_rows, best_strict, decompose


def _row(epoch, scope, *, agg, late, layered, phase):
    return {
        "event": "epoch",
        "epoch": epoch,
        "validation_scope": scope,
        "metrics": {
            "aggregate_relative_l2": agg,
            "time_bin_relative_l2": {"early": 0.1, "middle": 0.15, "late": late, "pre_onset": 0.2},
            "family_relative_l2": {"uniform": 0.12, "layered": layered, "marmousi": 0.25},
            "phase_correlation": phase,
        },
    }


def _write(tmp_path, rows):
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_strict_rows_ignores_light_panel(tmp_path):
    path = _write(tmp_path, [
        _row(1, "fixed_panel", agg=0.20, late=0.20, layered=0.20, phase=0.85),  # light: must be ignored
        _row(4, "fixed_full_time_panel", agg=0.238, late=0.28, layered=0.26, phase=0.81),
        {"event": "optimizer_update", "epoch": 4},  # non-epoch: ignored
    ])
    rows = strict_rows(path)
    assert set(rows) == {4}                       # only the strict epoch row survives
    assert best_strict(path)[0] == 4


def test_aggregate_only_smoothing_is_flagged():
    # aggregate clears 0.240 but late/layered/phase all fail -> smoothing, not a pass
    v = evaluate_gate(_row(12, "x", agg=0.2398, late=0.345, layered=0.288, phase=0.797)["metrics"])
    assert not v["passed"]
    assert v["aggregate_only_smoothing"]


def test_full_transport_completion_passes():
    v = evaluate_gate(_row(12, "x", agg=0.235, late=0.29, layered=0.26, phase=0.81)["metrics"])
    assert v["passed"]
    assert not v["aggregate_only_smoothing"]


def test_aggregate_fail_is_not_smoothing():
    # aggregate itself fails -> not the aggregate-only-smoothing signature
    v = evaluate_gate(_row(12, "x", agg=0.245, late=0.29, layered=0.26, phase=0.81)["metrics"])
    assert not v["passed"]
    assert not v["aggregate_only_smoothing"]


def test_decompose_reports_capped_regions():
    m = _row(12, "x", agg=0.24, late=0.345, layered=0.288, phase=0.80)["metrics"]
    m["coarse_metrics"] = {
        "aggregate_relative_l2": 0.30,
        "time_bin_relative_l2": {"early": 0.25, "middle": 0.29, "late": 0.41, "pre_onset": 0.68},
        "family_relative_l2": {"uniform": 0.21, "layered": 0.325, "marmousi": 0.33},
    }
    dec = decompose(m)
    # late-tbin barely improves (correction capped there); uniform improves a lot
    assert dec["tbin:late"]["improvement_frac"] < 0.20
    assert dec["family:uniform"]["improvement_frac"] > 0.20
    assert dec["aggregate"]["coarse"] == 0.30 and dec["aggregate"]["final"] == 0.24


def test_decompose_none_without_coarse():
    assert decompose(_row(12, "x", agg=0.24, late=0.3, layered=0.26, phase=0.8)["metrics"]) is None

