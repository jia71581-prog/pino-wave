#!/usr/bin/env python3
"""Select one passing B2-v10 observation arm by the frozen common-tail rule."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ARMS = ("peak", "cycle025", "cycle050", "fixed24")


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


def select_arm(arms: dict[str, dict]) -> dict:
    passing = [name for name in ARMS if bool(arms[name]["passed"])]
    if not passing:
        return {"status": "rejected", "selected": None}
    name = min(
        passing,
        key=lambda arm: (
            arms[arm]["adapted_common_tail_aggregate"],
            arms[arm]["observed_count"]["mean"],
            arm,
        ),
    )
    return {
        "status": "passed",
        "selected": {
            "arm": name,
            "adapted_aggregate": arms[name]["adapted_aggregate"],
            "parent_aggregate": arms[name]["parent_aggregate"],
            "adapted_common_tail_aggregate": arms[name]["adapted_common_tail_aggregate"],
            "parent_common_tail_aggregate": arms[name]["parent_common_tail_aggregate"],
            "mean_observed_count": arms[name]["observed_count"]["mean"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite arm decision: {args.output}")
    supervisor = Path(f"results/b2_v10_{args.tag}_four_arm_supervisor/terminal.json")
    if not supervisor.is_file():
        raise RuntimeError("four-arm supervisor has no terminal record")
    supervisor_payload = json.loads(supervisor.read_text())
    if supervisor_payload.get("status") != "complete":
        raise RuntimeError("four-arm supervisor did not complete")
    arms = {}
    bindings = {}
    for arm in ARMS:
        directory = Path(f"results/b2_v10_{args.tag}_{arm}")
        terminal = directory / "terminal.json"
        summary = directory / "summary.json"
        if not terminal.is_file() or not summary.is_file():
            raise RuntimeError(f"missing arm artifacts: {arm}")
        terminal_payload = json.loads(terminal.read_text())
        summary_payload = json.loads(summary.read_text())
        if terminal_payload.get("status") not in ("passed", "rejected"):
            raise RuntimeError(f"invalid arm terminal state: {arm}")
        arms[arm] = {
            key: summary_payload[key]
            for key in (
                "passed",
                "parent_aggregate",
                "adapted_aggregate",
                "parent_common_tail_aggregate",
                "adapted_common_tail_aggregate",
                "observed_count",
                "nonworse",
                "per_family",
                "runtime_s",
            )
        }
        bindings[arm] = {
            "summary_sha256": _sha256(summary),
            "terminal_sha256": _sha256(terminal),
        }
    decision = select_arm(arms)
    payload = {
        "schema": "b2_v10_calibration_arm_selection_v1",
        "selection_rule": "passing arm with lowest adapted common-tail aggregate, then smaller mean observed_count, then lexical name",
        "preregistration": str(args.preregistration),
        "preregistration_sha256": _sha256(args.preregistration),
        "supervisor_terminal": str(supervisor),
        "supervisor_terminal_sha256": _sha256(supervisor),
        "selector_sha256": _sha256(Path(__file__)),
        "arms": arms,
        "bindings": bindings,
        **decision,
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if decision["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
