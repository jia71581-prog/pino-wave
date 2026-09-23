"""Sealed parent identity and guarded promotion for V3 curriculum stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES
from .checkpoint import CHECKPOINT_FORMAT


@dataclass(frozen=True)
class CurriculumParentIdentity:
    manifest_digest: str
    pilot_run_digest: str
    parent_epoch: int
    parent_global_step: int
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    parent_validation_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StageDecision:
    accepted: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"accepted": self.accepted, "failures": list(self.failures)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_curriculum_parent(
    terminal_report: Mapping[str, object],
    best_validation: Mapping[str, object],
    best_checkpoint: str | Path,
    *,
    expected_manifest_digest: str,
    expected_run_digest: str,
) -> CurriculumParentIdentity:
    if terminal_report.get("status") != "complete":
        raise ValueError("curriculum parent pilot is not complete")
    if terminal_report.get("run_digest") != expected_run_digest:
        raise ValueError("curriculum parent terminal run digest mismatch")
    if best_validation.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("curriculum parent validation manifest mismatch")
    if best_validation.get("run_digest") != expected_run_digest:
        raise ValueError("curriculum parent validation run mismatch")
    epoch = int(best_validation.get("epoch", -1))
    global_step = int(best_validation.get("global_step", -1))
    terminal_epochs = int(terminal_report.get("epochs", -1))
    terminal_step = int(terminal_report.get("global_step", -1))
    if epoch <= 0 or epoch > terminal_epochs or global_step <= 0 or global_step > terminal_step:
        raise ValueError("curriculum parent validation epoch or step is invalid")
    validation = best_validation.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("curriculum parent validation metrics are missing")
    score = float(validation.get("score", math.nan))
    terminal_best = float(terminal_report.get("best_validation_score", math.nan))
    if not math.isfinite(score) or not math.isfinite(terminal_best) or not math.isclose(
        score, terminal_best, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError("curriculum parent best validation score mismatch")
    checkpoint = Path(best_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError("curriculum parent checkpoint cannot be loaded") from error
    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("curriculum parent checkpoint format mismatch")
    if payload.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("curriculum parent checkpoint manifest mismatch")
    if payload.get("config_digest") != expected_run_digest:
        raise ValueError("curriculum parent checkpoint run mismatch")
    if int(payload.get("epoch", -1)) != epoch or int(payload.get("global_step", -1)) != global_step:
        raise ValueError("curriculum parent checkpoint epoch or step mismatch")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or not math.isclose(
        float(metrics.get("validation_score", math.nan)), score, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError("curriculum parent checkpoint validation score mismatch")
    return CurriculumParentIdentity(
        manifest_digest=expected_manifest_digest,
        pilot_run_digest=expected_run_digest,
        parent_epoch=epoch,
        parent_global_step=global_step,
        parent_checkpoint_path=str(checkpoint),
        parent_checkpoint_sha256=_sha256(checkpoint),
        parent_validation_score=score,
    )


def curriculum_run_digest(
    config_digest: str,
    parent: CurriculumParentIdentity,
) -> str:
    if not config_digest:
        raise ValueError("curriculum config digest is required")
    encoded = json.dumps(
        {"config_digest": config_digest, "parent": parent.to_dict()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def accept_stage_checkpoint(
    *,
    parent_family_scores: Mapping[str, float],
    candidate_family_scores: Mapping[str, float],
    target_family: str,
    learned_families: Sequence[str],
    aggregate_parent: float,
    aggregate_candidate: float,
    aggregate_tolerance: float = 0.05,
    replay_tolerance: float = 0.10,
) -> StageDecision:
    allowed = set(ALLOWED_MEDIUM_TYPES)
    if set(parent_family_scores) != allowed or set(candidate_family_scores) != allowed:
        raise ValueError("stage family scores must contain exactly the three allowed families")
    if target_family not in allowed or set(learned_families) - allowed:
        raise ValueError("stage target or learned family is invalid")
    if min(aggregate_tolerance, replay_tolerance) < 0:
        raise ValueError("stage regression tolerances must be nonnegative")
    numeric = [
        float(aggregate_parent),
        float(aggregate_candidate),
        *(float(value) for value in parent_family_scores.values()),
        *(float(value) for value in candidate_family_scores.values()),
    ]
    if not all(math.isfinite(value) and value >= 0 for value in numeric):
        return StageDecision(False, ("stage metrics contain nonfinite or negative values",))
    failures: list[str] = []
    if float(candidate_family_scores[target_family]) >= float(parent_family_scores[target_family]):
        failures.append(f"target family {target_family} did not improve")
    if float(aggregate_candidate) > float(aggregate_parent) * (1.0 + aggregate_tolerance):
        failures.append("aggregate balanced score exceeded the regression tolerance")
    for family in learned_families:
        if float(candidate_family_scores[family]) > float(parent_family_scores[family]) * (
            1.0 + replay_tolerance
        ):
            failures.append(f"forgetting tolerance exceeded for {family}")
    return StageDecision(not failures, tuple(failures))


__all__ = [
    "CurriculumParentIdentity",
    "StageDecision",
    "accept_stage_checkpoint",
    "curriculum_run_digest",
    "validate_curriculum_parent",
]
