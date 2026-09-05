#!/usr/bin/env python3
"""Apply the frozen V9 two-seed gate and select one parent deterministically."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


VARIANTS = ("physics_cond", "spectral")
SEEDS = (372, 733)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def select_parent(lanes: dict[str, dict]) -> dict:
    """Return the frozen minimax-variant, best-seed decision."""
    variants = {}
    for variant in VARIANTS:
        rows = [lanes[f"{variant}_s{seed}"] for seed in SEEDS]
        passed = all(row["best_aggregate"] < row["baseline_aggregate"] for row in rows)
        variants[variant] = {
            "passed": passed,
            "worst_seed_best_aggregate": max(row["best_aggregate"] for row in rows),
            "mean_seed_best_aggregate": sum(row["best_aggregate"] for row in rows)
            / len(rows),
        }
    passing = [variant for variant in VARIANTS if variants[variant]["passed"]]
    if not passing:
        return {"status": "rejected", "variants": variants, "selected": None}
    variant = min(
        passing,
        key=lambda name: (
            variants[name]["worst_seed_best_aggregate"],
            variants[name]["mean_seed_best_aggregate"],
            name,
        ),
    )
    selected_key = min(
        (f"{variant}_s{seed}" for seed in SEEDS),
        key=lambda key: (lanes[key]["best_aggregate"], key),
    )
    return {
        "status": "passed",
        "variants": variants,
        "selected": {
            "lane": selected_key,
            "variant": variant,
            "seed": lanes[selected_key]["seed"],
            "checkpoint": lanes[selected_key]["checkpoint"],
            "checkpoint_sha256": lanes[selected_key]["checkpoint_sha256"],
            "best_aggregate": lanes[selected_key]["best_aggregate"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("results/b2_v9_parent_full_preregistration_20260901.json"),
    )
    parser.add_argument(
        "--supervisor-dir",
        type=Path,
        default=Path("results/b2_v9_parent_full_supervisor_20260901"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite decision: {args.output}")
    supervisor_terminal = args.supervisor_dir / "terminal.json"
    if not supervisor_terminal.is_file():
        raise RuntimeError("V9 supervisor has no terminal record")
    terminal = json.loads(supervisor_terminal.read_text())
    if terminal.get("status") != "complete" or terminal.get("exit_codes") != [0, 0, 0, 0]:
        raise RuntimeError("V9 supervisor did not complete all four lanes")
    prereg = json.loads(args.preregistration.read_text())
    lanes = {}
    for lane in prereg["lanes"]:
        variant = str(lane["arm"])
        seed = int(lane["seed"])
        key = f"{variant}_s{seed}"
        output_dir = Path(f"results/b2_v9_parent_{variant}_s{seed}_20260901")
        lane_terminal = output_dir / "terminal.json"
        checkpoint = output_dir / "best.pt"
        for required in (lane_terminal, output_dir / "best.json", output_dir / "run_identity.json", checkpoint):
            if not required.is_file():
                raise RuntimeError(f"missing V9 lane artifact: {required}")
        lane_terminal_payload = json.loads(lane_terminal.read_text())
        if lane_terminal_payload.get("status") != "complete":
            raise RuntimeError(f"V9 lane not complete: {key}")
        best = json.loads((output_dir / "best.json").read_text())
        identity = json.loads((output_dir / "run_identity.json").read_text())
        lanes[key] = {
            "variant": variant,
            "seed": seed,
            "baseline_aggregate": float(identity["baseline"]["aggregate"]),
            "best_aggregate": float(best["aggregate"]),
            "best_epoch": int(best["epoch"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "terminal_sha256": _sha256(lane_terminal),
            "best_json_sha256": _sha256(output_dir / "best.json"),
            "run_identity_sha256": _sha256(output_dir / "run_identity.json"),
        }
    decision = select_parent(lanes)
    payload = {
        "schema": "b2_v9_parent_selection_v1",
        "selection_rule": {
            "variant_gate": "both seed best aggregates strictly below their same-cache warp-anchor aggregate",
            "variant_choice": "smallest worst-seed best aggregate, then smallest mean, then lexical name",
            "seed_choice": "smallest best aggregate, then lexical lane key",
        },
        "preregistration": str(args.preregistration),
        "preregistration_sha256": _sha256(args.preregistration),
        "supervisor_terminal": str(supervisor_terminal),
        "supervisor_terminal_sha256": _sha256(supervisor_terminal),
        "selector_sha256": _sha256(Path(__file__)),
        "lanes": lanes,
        **decision,
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if decision["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
