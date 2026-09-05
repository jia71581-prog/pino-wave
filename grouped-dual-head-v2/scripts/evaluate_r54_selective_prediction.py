"""Apply the frozen R55 abstention rule to R54 holdout results.

Reads the frozen detector (coefficients, standardization, threshold, conformal
margin) from the R55 probe JSON and the per-record candidate errors from an R54
training run's best.json.  Recomputes the 14 deployable features from the
holdout cache, scores each record, and reports parent vs candidate coverage
under both frozen rules.  The abstention decision uses parent-side features
only, so it is identical for parent and candidate and remains deployable.

No refitting happens here: everything about the rule is read from the R55
artifact.  CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import importlib.util
import sys

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "r55_probe", _HERE / "analyze_r55_tailchain_abstention_probe.py"
)
r55 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = r55
_SPEC.loader.exec_module(r55)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coverage(abstained: np.ndarray, errors: np.ndarray, gate: float) -> dict:
    kept = errors[~abstained]
    return {
        "abstention_rate": float(np.mean(abstained)) if len(errors) else float("nan"),
        "kept_count": int(len(kept)),
        "kept_mean": float(np.mean(kept)) if len(kept) else float("nan"),
        "kept_max": float(np.max(kept)) if len(kept) else float("nan"),
        "kept_max_lte_gate": bool(len(kept) and float(np.max(kept)) <= gate),
        "missed_over_gate": int(np.sum(kept > gate)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--run-best-json", type=Path, required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    probe = json.loads(args.probe_json.read_text(encoding="utf-8"))
    gate = float(probe["gate"])
    detector = probe["detector"]
    names = probe["feature_names"]
    mean = np.asarray(detector["feature_mean"], dtype=np.float64)
    std = np.asarray(detector["feature_std"], dtype=np.float64)
    weights = np.asarray(
        [detector["coefficients"][name] for name in names], dtype=np.float64
    )
    intercept = float(detector["intercept"])
    threshold = float(probe["fit_only_threshold"]["threshold"])
    margin = float(probe["fit_only_conformal"]["margin"])

    run = json.loads(args.run_best_json.read_text(encoding="utf-8"))
    records = run["metrics"]["records"]
    by_id = {row["sample_id"]: row for row in records}

    import h5py

    sample_ids, features = [], []
    for path in args.holdout_cache:
        with h5py.File(path, "r") as handle:
            for index in range(handle["family"].shape[0]):
                raw = handle["sample_id"][index]
                sample_ids.append(raw.decode() if isinstance(raw, bytes) else str(raw))
                features.append(r55.record_features(handle, index))
    features = np.asarray(features)

    missing = [s for s in sample_ids if s not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} cache records missing from run: {missing[:5]}")

    scores = ((features - mean) / std) @ weights + intercept
    predicted = np.exp(scores)
    parent = np.asarray([by_id[s]["parent_rel_l2"] for s in sample_ids])
    candidate = np.asarray([by_id[s]["candidate_rel_l2"] for s in sample_ids])
    families = [by_id[s]["family"] for s in sample_ids]

    abstain_threshold = scores >= threshold
    abstain_conformal = predicted * margin >= gate

    def rule_block(abstained: np.ndarray) -> dict:
        block = {
            "parent": coverage(abstained, parent, gate),
            "candidate": coverage(abstained, candidate, gate),
            "per_family_candidate": {},
        }
        for family in sorted(set(families)):
            mask = np.asarray([f == family for f in families])
            block["per_family_candidate"][family] = coverage(
                abstained[mask], candidate[mask], gate
            )
        return block

    kept_conf = ~abstain_conformal
    payload = {
        "schema": "r54_selective_prediction_evaluation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": gate,
        "rule_source": {
            "probe_json": str(args.probe_json.resolve()),
            "probe_sha256": sha256(args.probe_json),
            "frozen_threshold": threshold,
            "frozen_conformal_margin": margin,
            "refit_performed": False,
        },
        "run_source": {
            "best_json": str(args.run_best_json.resolve()),
            "best_json_sha256": sha256(args.run_best_json),
            "epoch": run.get("epoch"),
            "checkpoint_sha256": run.get("checkpoint_sha256"),
        },
        "counts": {"holdout_records": int(len(sample_ids))},
        "aggregate_no_abstention": {
            "parent_mean": float(np.mean(parent)),
            "parent_max": float(np.max(parent)),
            "candidate_mean": float(np.mean(candidate)),
            "candidate_max": float(np.max(candidate)),
        },
        "fit_only_threshold": rule_block(abstain_threshold),
        "fit_only_conformal": rule_block(abstain_conformal),
        "kept_records_conformal": [
            {
                "sample_id": sample_ids[i],
                "family": families[i],
                "predicted_rel_l2": float(predicted[i]),
                "parent_rel_l2": float(parent[i]),
                "candidate_rel_l2": float(candidate[i]),
            }
            for i in np.flatnonzero(kept_conf)
        ],
        "data_boundary": {
            "validation_opened": False,
            "test_id_opened": False,
            "development_holdout_only": True,
        },
        "evidence": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "holdout_cache": [str(p.resolve()) for p in args.holdout_cache],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
