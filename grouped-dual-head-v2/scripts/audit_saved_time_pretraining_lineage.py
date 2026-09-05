#!/usr/bin/env python3
"""Read-only, bounded leakage audit for Saved-Time pretraining lineages.

The audit never opens a checkpoint tensor or probes a process.  It follows only
``parent_checkpoint_identity`` links, reports configuration-level uses of
validation truth, and records enough file identity information to make a later
train-only experiment fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


def _sha256(path: Path, *, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _resolve_link(value: object, *, identity_path: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = identity_path.parent / path
    return path.resolve()


def config_leakage_findings(config: Mapping[str, object]) -> list[dict[str, object]]:
    """Return conservative, machine-readable leakage findings for one config."""

    findings: list[dict[str, object]] = []
    control = config.get("epoch_validation_control") or {}
    if isinstance(control, Mapping) and bool(control.get("enabled", False)):
        split = str(control.get("evaluation_split", "validation"))
        if split != "train":
            findings.append(
                {
                    "code": "epoch_control_uses_nontrain_truth",
                    "evaluation_split": split,
                }
            )

    adaptive = config.get("adaptive_sampling") or {}
    if isinstance(adaptive, Mapping) and bool(adaptive.get("enabled", False)):
        explicit = any(
            adaptive.get(name) is not None
            for name in ("time_bin_errors", "family_errors")
        )
        evidence_split = adaptive.get("evidence_split")
        if explicit and str(evidence_split) != "train":
            findings.append(
                {
                    "code": "adaptive_errors_lack_train_evidence_binding",
                    "evidence_split": evidence_split,
                }
            )

    parent_checkpoint = config.get("parent_checkpoint")
    if parent_checkpoint not in (None, ""):
        checkpoint_name = Path(str(parent_checkpoint)).name
        if checkpoint_name == "latest.pt":
            findings.append({"code": "mutable_parent_checkpoint_path"})
        if (
            checkpoint_name == "best.pt"
            and str(config.get("parent_checkpoint_selection_split")) != "train"
        ):
            findings.append(
                {
                    "code": "parent_best_checkpoint_lacks_train_selection_binding",
                    "selection_split": config.get(
                        "parent_checkpoint_selection_split"
                    ),
                }
            )
    return findings


def audit_lineage(
    identity_path: str | Path,
    *,
    max_depth: int = 16,
    hash_checkpoints: bool = False,
) -> dict[str, object]:
    """Follow a single parent chain without unbounded filesystem discovery."""

    limit = int(max_depth)
    if limit <= 0:
        raise ValueError("max_depth must be positive")
    current = Path(identity_path).expanduser().resolve()
    seen: set[Path] = set()
    nodes: list[dict[str, object]] = []
    terminal = "no_parent_identity"

    for depth in range(limit):
        if current in seen:
            terminal = "cycle"
            break
        seen.add(current)
        if not current.is_file():
            nodes.append(
                {
                    "depth": depth,
                    "identity_path": str(current),
                    "identity_exists": False,
                    "findings": [{"code": "missing_parent_identity"}],
                }
            )
            terminal = "missing_parent_identity"
            break

        payload = json.loads(current.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"identity is not a mapping: {current}")
        config = payload.get("config") or {}
        if not isinstance(config, Mapping):
            raise ValueError(f"identity config is not a mapping: {current}")
        checkpoint = _resolve_link(
            config.get("parent_checkpoint"), identity_path=current
        )
        checkpoint_record: dict[str, object] | None = None
        if checkpoint is not None:
            exists = checkpoint.is_file()
            checkpoint_record = {
                "path": str(checkpoint),
                "exists": exists,
                "size_bytes": checkpoint.stat().st_size if exists else None,
            }
            if exists and hash_checkpoints:
                checkpoint_record["sha256"] = _sha256(checkpoint)

        findings = config_leakage_findings(config)
        if checkpoint_record is not None and not bool(checkpoint_record["exists"]):
            findings.append({"code": "missing_parent_checkpoint"})
        nodes.append(
            {
                "depth": depth,
                "identity_path": str(current),
                "identity_exists": True,
                "identity_sha256": _sha256(current),
                "run_digest": payload.get("run_digest"),
                "manifest_digest": payload.get("manifest_digest"),
                "artifact_dir": config.get("artifact_dir"),
                "parent_checkpoint": checkpoint_record,
                "findings": findings,
            }
        )
        parent = _resolve_link(
            config.get("parent_checkpoint_identity"), identity_path=current
        )
        if parent is None:
            terminal = "no_parent_identity"
            break
        current = parent
    else:
        terminal = "max_depth_reached"

    finding_count = sum(len(node["findings"]) for node in nodes)
    return {
        "schema": "saved_time_pretraining_lineage_audit_v1",
        "root_identity": str(Path(identity_path).expanduser().resolve()),
        "max_depth": limit,
        "terminal": terminal,
        "node_count": len(nodes),
        "finding_count": finding_count,
        "promotion_safe_from_config_evidence": finding_count == 0
        and terminal == "no_parent_identity",
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument(
        "--hash-checkpoints",
        action="store_true",
        help="also SHA-256 checkpoint bytes; disabled by default to keep the audit cheap",
    )
    args = parser.parse_args()
    report = audit_lineage(
        args.identity,
        max_depth=args.max_depth,
        hash_checkpoints=args.hash_checkpoints,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["promotion_safe_from_config_evidence"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
