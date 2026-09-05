#!/usr/bin/env python3
"""V15 CLI: mask-confined correction candidate, implementation and gate wiring.

This session's authorization covers implementation and unit tests only.  No
stage may be launched from this script without a fresh lead authorization file:
``smoke``, ``pilot``, ``scale-*``, ``long``, ``final-train-confirm``,
``validation-once`` and ``test-once`` all refuse by default.

Inherited vetoes enforced in this module:

(a) thresholds are transcribed from the frozen preregistration and never moved
    toward a measured value;
(b) the correction mask is derived from parent frame energy only, never from
    deployment-time truth, including truth frame energy;
(c) the three v14 oracle constants are void; no transplanted or synthetic-panel
    constant is an acceptance baseline;
(d) the smoke loss gate is a wiring check and is never learning or convergence
    evidence;
(e) the correction is an output-field data fit plus a feature-driven ansatz; it
    is not a physics or PDE residual and not deployment-time truth supervision,
    and no PDE residual is evaluated anywhere in this pipeline;
(f) smoke evidence never promotes to pilot or long.

Acceptance is convention (iii) ``global_energy_rel_l2`` at the inherited 0.05
gate.  Convention (ii) is a training proxy and carries no acceptance claim.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import (
    AUTH_SCHEMA,
    AccessLedger,
    Lineage,
    canonical_sha,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import CandidateSpool
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v14 import (
    build_test_terminal,
    build_validation_terminal,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v15 import (
    CANDIDATE,
    LOSS_REDUCTION_MIN,
    ORACLE_GAIN_FRACTION,
    SMOKE_DISCLAIMERS,
    VOIDED_ORACLE_PROVENANCE,
    V15ProductionBackend,
    complete_identity_v15,
    record_oracle_bound,
    refuse_promotion_from_smoke,
    run_v15_smoke,
    terminal_v15,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import (
    BASIS_FILE_SHA256,
    BASIS_TENSOR_SHA256,
    PANELS_SHA256,
    PARENT_PATH,
    PARENT_SHA256,
    BindingRefusal,
    atomic_json_exclusive,
    sha256_file,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v3 import (
    ACCEPTANCE_CONVENTION,
    TAU,
    TRAINING_PROXY_CONVENTION,
    CudaPeakVram,
    coefficient_energy_definition,
    loss_specification,
    mask_provenance,
)
import scripts.train_r16_dscp_v14 as v14


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
ENGINE = ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v15.py"
HARNESS = ROOT / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v3.py"
TEST = ROOT / "tests/saved_time_phase_operator_v4/test_r16_dscp_v15.py"
CONFIG = ROOT / "configs/r16_dscp_v15.yaml"
OUT = ROOT / "results/r16_dscp_v15"
STATIC = OUT / "static_evidence.json"
PREREG = ROOT / "results/r16_dscp_v15_preregistration_20260826.json"
PROBE_REFERENCE = ROOT / "scripts/probe_r4e9_mask_confined.py"
R4E9_TERMINAL = ROOT / "results/r4e9_mask_confined_oracle_20260826/terminal.json"
BASIS = ROOT / "results/r16_dscp_v1/basis_rank16.pt"
PANELS = ROOT / "results/r16_dscp_v1/panels.json"

#: bindings frozen by the lead for this candidate; drift is a hard stop
FROZEN_BINDINGS = {
    "preregistration": (
        PREREG,
        "a472ff436a9a0d1453d8ce8c8b664f0b1d674b24ba46796139e7a5ca1dd65d15",
    ),
    "r4e9_probe_reference": (
        PROBE_REFERENCE,
        "f19491548e4263d861a432835b2f8a5eb9b03c88727d944a71d02a57b9498d65",
    ),
    "r4e9_terminal_reference": (
        R4E9_TERMINAL,
        "410e4b14bbfb21a1a5fc896d0ddf857e8a6395a872c7217e19cfea4aa266e505",
    ),
    "parent_checkpoint": (
        Path(PARENT_PATH),
        "448035bd0061205c67799eeef3b023a71b49e2db7028b6155078be1a886de789",
    ),
}

SEALED_SPLITS = ("validation", "test_id")

MODES = (
    "prep",
    "verify-bindings",
    "authorize-stage",
    "smoke",
    "pilot",
    "scale-1",
    "scale-4",
    "scale-decide",
    "long",
    "final-train-confirm",
    "validation-once",
    "test-once",
)

STAGE_MODES = MODES[3:]

#: inherited v14 thresholds, transcribed and not modified
THRESHOLDS = {
    "joint_improvement": 0.01,
    "per_family_improvement": 0.005,
    "nonworse_records": 23,
    "late_high_nonworse": True,
    "e2e_parent_ratio_max": 1.05,
    "absolute_rel_l2_max": 0.05,
    "traditional_speedup_min": 10.0,
    "acceptance_convention": ACCEPTANCE_CONVENTION,
    "acceptance_convention_modified": False,
    "smoke_loss_reduction_min": LOSS_REDUCTION_MIN,
    "smoke_oracle_gain_fraction": ORACLE_GAIN_FRACTION,
}

INHERITED_VETOES = (
    "thresholds must not be moved toward measured values",
    "no deployment-time truth in the mask, including truth frame energy",
    "no cross-record or synthetic-sample oracle constant as an acceptance baseline",
    "the smoke loss gate is never learning or convergence evidence",
    "the correction is an output-field data fit plus a feature-driven ansatz, "
    "not a physics or PDE residual and not deployment-time truth supervision; "
    "zero PDE residual anywhere in the pipeline",
    "smoke evidence never promotes to pilot or long",
)

#: lead ruling 1 (2026-08-26): the smoke stage reuses the three r4e9 probe
#: records.  Recorded here so the choice and its stated reason travel with the
#: terminal artifact.
SMOKE_RECORD_DECISION = {
    "decided_by": "lead",
    "decided_on": "2026-08-26",
    "ruling": "smoke_records_reuse_the_r4e9_probe_records",
    "sample_ids": (
        "train_uniform_00321",
        "train_layered_00564",
        "train_marmousi_00385",
    ),
    "same_as_r4e9_probe_records": True,
    "panel_role": "smoke",
    "split": "train",
    "rationale": (
        "smoke is not learning or convergence evidence under inherited veto (d), "
        "so aligning it with the r4e9 probe records only makes diagnosis easier; "
        "it produces no promotable conclusion, therefore 'these records were seen "
        "by the probe' is not a contamination risk"
    ),
    "produces_promotable_conclusion": False,
}

#: lead ruling 2 (2026-08-26): the oracle keeps the parent-energy-mask
#: convention already implemented in ``confined_oracle_upper_bound``.
SMOKE_ORACLE_DECISION = {
    "decided_by": "lead",
    "decided_on": "2026-08-26",
    "ruling": "keep_the_parent_energy_mask_convention_of_confined_oracle_upper_bound",
    "implementation": (
        "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v3.py"
        "::confined_oracle_upper_bound"
    ),
    "mask_source": "parent_frame_energy_only",
    "truth_split_read": "train",
    "sealed_splits_read": [],
    "deviation_from_r4e9_probe_retained": True,
    "is_achieved_result": False,
    "recomputed_on_the_scored_record": True,
    "transplanted_constants_used": False,
    "synthetic_records_used": False,
}


class V15BindingRefusal(BindingRefusal):
    """A frozen v15 binding is absent or has drifted."""


class StageAuthorizationRequired(BindingRefusal):
    """A stage was requested without a fresh lead authorization."""


class SealedSplitRefusal(BindingRefusal):
    """A sealed future wavefield was approached without a frozen authorization."""


# ---------------------------------------------------------------------------
# bindings
# ---------------------------------------------------------------------------


def bind(path: str | Path) -> dict[str, Any]:
    item = Path(path).resolve()
    if not item.exists():
        raise V15BindingRefusal(f"binding target absent: {item}")
    return {
        "path": str(item),
        "sha256": sha256_file(item),
        "size_bytes": item.stat().st_size,
    }


def verify_bindings(*, require_parent: bool = True) -> dict[str, Any]:
    """Recheck every frozen binding.  Any drift is a refusal, never a warning."""
    report: dict[str, Any] = {"candidate": CANDIDATE, "checked": {}, "drift": []}
    for name, (path, expected) in FROZEN_BINDINGS.items():
        item = Path(path)
        if not item.exists():
            row = {
                "path": str(item),
                "expected_sha256": expected,
                "observed_sha256": None,
                "status": "absent",
            }
            report["checked"][name] = row
            if name == "parent_checkpoint" and not require_parent:
                row["status"] = "measurement_gap"
                continue
            report["drift"].append(name)
            continue
        observed = sha256_file(item)
        row = {
            "path": str(item),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "size_bytes": item.stat().st_size,
            "status": "match" if observed == expected else "drift",
        }
        report["checked"][name] = row
        if observed != expected:
            report["drift"].append(name)
    report["mutable_now"] = {
        "engine": bind(ENGINE) if ENGINE.exists() else None,
        "harness": bind(HARNESS) if HARNESS.exists() else None,
        "script": bind(SCRIPT),
        "config": bind(CONFIG) if CONFIG.exists() else None,
        "test": bind(TEST) if TEST.exists() else None,
    }
    report["passed"] = not report["drift"]
    if report["drift"]:
        raise V15BindingRefusal(f"binding drift: {sorted(report['drift'])}")
    return report


def parent_write_audit() -> dict[str, Any]:
    """Audit the parent checkpoint: it must be read-only for this candidate.

    If the path is unreachable from this harness the audit says so plainly and
    reports a measurement gap rather than inventing a zero.
    """
    path = Path(PARENT_PATH)
    if not path.exists():
        # an unreachable audit is reported as a measurement gap, never worked around
        return {
            "path": str(path),
            "reachable": False,
            "writes": None,
            "status": "measurement_gap",
            "note": "parent checkpoint not reachable from this harness; not worked around",
        }
    stat = path.stat()
    observed = sha256_file(path)
    expected = FROZEN_BINDINGS["parent_checkpoint"][1]
    return {
        "path": str(path),
        "reachable": True,
        "sha256": observed,
        "expected_sha256": expected,
        "sha256_matches_frozen": observed == expected,
        "declared_sha256_in_harness": PARENT_SHA256,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "writes": 0 if observed == expected else None,
        "status": "read_only_unchanged" if observed == expected else "drift",
        "opened_for_write_by_v15": False,
    }


def sealed_split_guard(split: str) -> None:
    """Refuse any approach to sealed future wavefields during development."""
    if str(split) in SEALED_SPLITS:
        raise SealedSplitRefusal(
            f"{split} future wavefields are sealed during algorithm development; "
            "a frozen sealed-stage authorization is required"
        )


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------

REQUIRED_AUTHORIZATION_KEYS = (
    "candidate",
    "stage",
    "preregistration_sha256",
    "lead_authorized",
    "acknowledges_post_hoc_rule_revision",
    "acknowledges_inherited_vetoes",
)


def validate_stage_authorization(payload: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """A stage runs only against a complete, current, lead-signed authorization."""
    missing = [key for key in REQUIRED_AUTHORIZATION_KEYS if key not in payload]
    if missing:
        raise StageAuthorizationRequired(f"authorization missing keys: {missing}")
    if str(payload["candidate"]) != CANDIDATE:
        raise StageAuthorizationRequired("authorization is for another candidate")
    if str(payload["stage"]) != str(stage):
        raise StageAuthorizationRequired(
            f"authorization is for stage {payload['stage']}, not {stage}"
        )
    if not payload["lead_authorized"]:
        raise StageAuthorizationRequired("authorization is not lead authorized")
    if not payload["acknowledges_post_hoc_rule_revision"]:
        raise StageAuthorizationRequired(
            "authorization must acknowledge the recorded post-hoc rule revision"
        )
    if not payload["acknowledges_inherited_vetoes"]:
        raise StageAuthorizationRequired("authorization must acknowledge inherited vetoes")
    frozen = FROZEN_BINDINGS["preregistration"][1]
    if str(payload["preregistration_sha256"]) != frozen:
        raise StageAuthorizationRequired("authorization binds a different preregistration")
    if stage in {"pilot", "long"} and payload.get("promoted_from") == "smoke":
        refuse_promotion_from_smoke(stage)
    if stage in {"validation-once", "test-once"} and not payload.get("sealed_chain_verified"):
        raise SealedSplitRefusal(
            "sealed evaluation requires a verified final-train-confirm chain"
        )
    return dict(payload)


def load_stage_authorization(path: str | Path, stage: str) -> dict[str, Any]:
    item = Path(path)
    if not item.is_file():
        raise StageAuthorizationRequired(f"no stage authorization at {item}")
    payload = json.loads(item.read_text())
    validated = validate_stage_authorization(payload, stage)
    validated["authorization_path"] = str(item.resolve())
    validated["authorization_sha256"] = sha256_file(item)
    return validated


def authorize_stage(
    *,
    stage: str,
    output: str | Path,
    lead_authorized: bool,
    acknowledges_post_hoc_rule_revision: bool,
    acknowledges_inherited_vetoes: bool,
    input_checkpoint: str | None = None,
    sealed_chain_verified: bool = False,
) -> dict[str, Any]:
    """Write a stage authorization.  Bindings are rechecked before it is issued."""
    if stage not in STAGE_MODES:
        raise StageAuthorizationRequired(f"unknown stage {stage}")
    bindings = verify_bindings(require_parent=Path(PARENT_PATH).exists())
    payload = {
        "schema": "r16_dscp_v15_stage_authorization_v1",
        "candidate": CANDIDATE,
        "stage": stage,
        "issued_at": time.time(),
        "preregistration_path": str(PREREG),
        "preregistration_sha256": FROZEN_BINDINGS["preregistration"][1],
        "lead_authorized": bool(lead_authorized),
        "acknowledges_post_hoc_rule_revision": bool(acknowledges_post_hoc_rule_revision),
        "acknowledges_inherited_vetoes": bool(acknowledges_inherited_vetoes),
        "inherited_vetoes": list(INHERITED_VETOES),
        "sealed_chain_verified": bool(sealed_chain_verified),
        "input_checkpoint_path": None if input_checkpoint is None else str(Path(input_checkpoint).resolve()),
        "bindings": bindings["checked"],
        "thresholds": dict(THRESHOLDS),
        "disclaimers": dict(SMOKE_DISCLAIMERS),
    }
    payload["authorization_digest"] = canonical_sha(payload)
    validate_stage_authorization(payload, stage)
    atomic_json_exclusive(payload, Path(output))
    return payload


# ---------------------------------------------------------------------------
# static evidence, including the recorded fix-4 choice and rationale
# ---------------------------------------------------------------------------


def static_evidence() -> dict[str, Any]:
    """The v15 terminal artifact for the implementation stage."""
    config = yaml.safe_load(CONFIG.read_text()) if CONFIG.exists() else {}
    return {
        "schema": "r16_dscp_v15_static_evidence_v1",
        "candidate": CANDIDATE,
        "status": "implementation_only",
        "stages_run": [],
        "gpu_training_started": False,
        "bindings": verify_bindings(require_parent=Path(PARENT_PATH).exists())["checked"],
        "parent_write_audit": parent_write_audit(),
        "files": {
            "script": bind(SCRIPT),
            "engine": bind(ENGINE),
            "harness": bind(HARNESS),
            "config": bind(CONFIG),
            "test": bind(TEST) if TEST.exists() else None,
        },
        "mask": mask_provenance(tau=TAU),
        "loss_specification": loss_specification(),
        "coefficient_energy_dimensional_fix": coefficient_energy_definition(),
        "voided_v14_oracle_constants": dict(VOIDED_ORACLE_PROVENANCE),
        "v14_defects_fixed": {
            "mask_confined_correction": (
                "the correction is multiplied by the parent-energy keep mask "
                "before application; it is exactly zero outside the mask and "
                "the output there is bit-identical to the parent"
            ),
            "smoke_loss_reduction_per_record": (
                "PerRecordLossLedger keys losses by record; the v14 first/last "
                "of a round-robin stream is not expressible"
            ),
            "oracle_recomputed_on_scored_record": (
                "confined_oracle_upper_bound is evaluated on the record it "
                "scores; the three v14 constants are void"
            ),
            "coefficient_energy_dimensionless": (
                "normalized by the mean squared parent amplitude on the "
                "retained future frames"
            ),
            "peak_vram_measured": (
                "CudaPeakVram samples torch.cuda.max_memory_reserved on every "
                "cached path and the gate refuses an unmeasured or zero peak"
            ),
        },
        "acceptance": {
            "convention": ACCEPTANCE_CONVENTION,
            "validation_test_rel_l2_max": THRESHOLDS["absolute_rel_l2_max"],
            "modified": False,
            "training_proxy_convention": TRAINING_PROXY_CONVENTION,
            "training_proxy_may_support_acceptance": False,
        },
        "inherited_vetoes": list(INHERITED_VETOES),
        "disclaimers": dict(SMOKE_DISCLAIMERS),
        "sealed_splits": list(SEALED_SPLITS),
        "sealed_future_wavefields_opened": False,
        "config_digest": canonical_sha(config),
    }


def write_static_evidence(path: str | Path = STATIC) -> dict[str, Any]:
    payload = static_evidence()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_exclusive(payload, target)
    return payload


# ---------------------------------------------------------------------------
# backend construction
# ---------------------------------------------------------------------------


def promote_backend_to_v15(backend: Any, *, tau: float = TAU) -> Any:
    """Re-class a constructed v14 backend as a v15 backend.

    Mirrors the v14 script's own re-class step.  The v15 spool root, the VRAM
    monitor and the mask threshold are installed here.
    """
    backend.__class__ = V15ProductionBackend
    backend.spool = CandidateSpool(
        backend.run_dir,
        run_digest=backend.lineage.run_digest,
        rank=int(os.environ.get("RANK", "0")),
        owned_root="/dev/shm/r16_dscp_v15",
    )
    backend.tau = float(tau)
    backend.vram = CudaPeakVram(backend.device).reset()
    backend.mask_events = []
    backend._sample_vram()
    return backend


def build_v15_backend(
    mode: str,
    auth: Mapping[str, Any],
    line: Lineage,
    run: Path,
    identity: Mapping[str, Any],
    **kwargs: Any,
):
    bundle = v14.build_v14_backend(mode, auth, line, run, identity, **kwargs)
    return (promote_backend_to_v15(bundle[0]), *bundle[1:])


# ---------------------------------------------------------------------------
# production smoke factory
# ---------------------------------------------------------------------------


def v15_effective() -> dict[str, str]:
    """The effective binding set for this candidate.

    v15 has no ``design_preflight.json``; the run digest is derived from this
    mapping instead, so it moves if and only if a binding moves.
    """
    return {
        "code": sha256_file(ENGINE),
        "script": sha256_file(SCRIPT),
        "config": sha256_file(CONFIG),
        "harness": sha256_file(HARNESS),
        "panels": sha256_file(PANELS),
        "basis": sha256_file(BASIS),
        "parent": sha256_file(PARENT_PATH),
        "preregistration": FROZEN_BINDINGS["preregistration"][1],
    }


def v15_run_digest(effective: Mapping[str, str] | None = None) -> str:
    return canonical_sha(dict(effective if effective is not None else v15_effective()))


def build_smoke_lineage(run_digest: str) -> Lineage:
    return Lineage(
        CANDIDATE,
        "smoke",
        str(run_digest),
        sha256_file(ENGINE),
        sha256_file(CONFIG),
        PANELS_SHA256,
        BASIS_FILE_SHA256,
        BASIS_TENSOR_SHA256,
        PARENT_SHA256,
        "none",
    )


def build_smoke_access_authorization(
    stage_authorization: Mapping[str, Any], line: Lineage
) -> dict[str, Any]:
    """Synthesize the inner v3 data-access authorization for the smoke stage.

    The lead stage authorization carries the human decision; the v3 engine
    demands its own schema before any truth read.  Both are recorded, and this
    one is bound to the same lineage hashes the stage authorization checked.
    Only the train split is reachable through it.
    """
    payload = {
        "schema": AUTH_SCHEMA,
        "status": "authorized",
        **line.__dict__,
        "candidate": CANDIDATE,
        "effective": v15_effective(),
        "input_checkpoint_path": None,
        "input_run_identity": None,
        "split": "train",
        "sealed_splits_authorized": [],
        "preregistration_path": str(PREREG),
        "preregistration_sha256": FROZEN_BINDINGS["preregistration"][1],
        "stage_authorization_path": stage_authorization.get("authorization_path"),
        "stage_authorization_sha256": stage_authorization.get("authorization_sha256"),
        "gates_sha256": canonical_sha(yaml.safe_load(CONFIG.read_text())),
    }
    payload["authorization_digest"] = canonical_sha(payload)
    return payload


def build_smoke_oracle(backend: Any) -> Callable[[int, Mapping[str, Any]], Mapping[str, Any]]:
    """Per-record oracle bound, recomputed on the record being scored.

    Lead ruling 2: keep the ``confined_oracle_upper_bound`` convention, i.e. the
    fit restriction and the confinement gate both come from the PARENT energy
    mask.  Truth is read from the train split only, and the payload keeps its
    ``deviation_from_r4e9_probe`` note.  The result is an upper bound and never
    an achieved result.
    """

    def oracle(index: int, row: Mapping[str, Any]) -> Mapping[str, Any]:
        key = backend.index_keys[int(index)]
        entry = backend.cache.get(key, AccessLedger(backend.split))
        if int(entry.route_index) < 0:
            raise V15BindingRefusal(
                f"record {row.get('sample_id')} abstained; no basis exists for an "
                "oracle bound on it, and no other record's bound may stand in"
            )
        k1 = int(entry.public.observed_indices[1])
        payload = record_oracle_bound(
            sample_id=str(row["sample_id"]),
            family=str(row.get("family", backend.family_by_sample[key.sample_id])),
            parent_future=entry.parent[k1 + 1 :],
            truth_future=entry.truth,
            basis_future=backend.candidate.bases[int(entry.route_index)].detach().cpu()[k1 + 1 :],
            tau=float(backend.tau),
        )
        payload["decision"] = dict(SMOKE_ORACLE_DECISION)
        return payload

    return oracle


def build_smoke_terminal_writer(
    *,
    run: Path,
    stage_authorization: Mapping[str, Any],
    access_authorization: Mapping[str, Any],
    records: Mapping[str, Any],
) -> Callable[[Mapping[str, Any]], None]:
    """Write the smoke terminal once, with both lead rulings recorded.

    Nothing here touches a gate, a threshold or a metric definition; the payload
    is annotated and persisted exactly as the engine produced it.
    """

    def terminal(payload: Mapping[str, Any]) -> None:
        annotated = {
            **dict(payload),
            "smoke_record_decision": {**dict(SMOKE_RECORD_DECISION), **dict(records)},
            "oracle_decision": dict(SMOKE_ORACLE_DECISION),
            "stage_authorization": {
                "path": stage_authorization.get("authorization_path"),
                "sha256": stage_authorization.get("authorization_sha256"),
                "lead_authorized": bool(stage_authorization.get("lead_authorized")),
                "acknowledges_post_hoc_rule_revision": bool(
                    stage_authorization.get("acknowledges_post_hoc_rule_revision")
                ),
                "acknowledges_inherited_vetoes": bool(
                    stage_authorization.get("acknowledges_inherited_vetoes")
                ),
            },
            "access_authorization_digest": access_authorization["authorization_digest"],
            "bindings": verify_bindings(require_parent=True)["checked"],
            "parent_write_audit": parent_write_audit(),
            "inherited_vetoes": list(INHERITED_VETOES),
            "thresholds": dict(THRESHOLDS),
            "acceptance_convention_modified": False,
            "sealed_future_wavefields_opened": False,
            "sealed_splits": list(SEALED_SPLITS),
        }
        atomic_json_exclusive(annotated, run / "terminal.json")

    return terminal


def build_v15_smoke_bundle(
    stage_authorization: Mapping[str, Any], args: argparse.Namespace | None = None
) -> dict[str, Any]:
    """Assemble every argument :func:`run_v15_smoke` requires.

    The bindings are rechecked here, immediately before construction, so a file
    that moved between authorization load and factory build is a refusal.
    """
    verify_bindings(require_parent=True)
    audit_before = parent_write_audit()
    if audit_before["status"] != "read_only_unchanged":
        raise V15BindingRefusal(f"parent checkpoint audit refused: {audit_before['status']}")

    effective = v15_effective()
    run_digest = v15_run_digest(effective)
    line = build_smoke_lineage(run_digest)
    access = build_smoke_access_authorization(stage_authorization, line)
    identity = complete_identity_v15(
        mode="smoke",
        run_digest=run_digest,
        authorization_sha256=str(stage_authorization.get("authorization_sha256", "none")),
        engine_sha256=sha256_file(ENGINE),
        script_sha256=sha256_file(SCRIPT),
        config_sha256_effective=sha256_file(CONFIG),
        panels_sha256_effective=PANELS_SHA256,
        basis_sha256_effective=BASIS_FILE_SHA256,
        parent_sha256_effective=PARENT_SHA256,
        input_checkpoint_sha256="none",
    )

    run = OUT / "smoke"
    if run.exists():
        raise FileExistsError(f"smoke stage directory already exists: {run}")
    run.mkdir(parents=True)
    atomic_json_exclusive(identity, run / "run_identity.json")
    atomic_json_exclusive(access, run / "access_authorization.json")

    backend, roles, _all_indices, _metadata = build_v15_backend(
        "smoke", access, line, run, identity
    )
    if backend.split != "train":
        raise SealedSplitRefusal(f"smoke must run on train, got {backend.split}")
    sealed_split_guard(backend.split)

    records = list(roles["smoke"])
    expected = list(SMOKE_RECORD_DECISION["sample_ids"])
    panel_ids = [
        row["sample_id"]
        for row in json.loads(PANELS.read_text())["records"]
        if row.get("role") == "smoke"
    ]
    if panel_ids != expected:
        raise V15BindingRefusal(
            f"smoke panel records drifted from the lead ruling: {panel_ids} != {expected}"
        )
    if len(records) != len(expected):
        raise V15BindingRefusal("smoke role indices do not match the three ruled records")

    def record_key(index: int) -> str:
        return str(backend.index_keys[int(index)].sample_id)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    return {
        "backend": backend,
        "records": records,
        "record_key": record_key,
        "quick_gate": lambda rows: all(
            float(row["aggregate_rel_l2"]) <= float(row["parent_rel_l2"]) for row in rows
        ),
        "checkpoint": lambda: backend.checkpoint({"mode": "smoke"}),
        "resource_snapshot": lambda saved: backend.resources(saved, world),
        "terminal": build_smoke_terminal_writer(
            run=run,
            stage_authorization=stage_authorization,
            access_authorization=access,
            records={"indices": records, "resolved_sample_ids": expected},
        ),
        "oracle": build_smoke_oracle(backend),
        "lineage": backend.checkpoint_identity,
        "effective": effective,
        "parent": {
            **audit_before,
            "writes": 0,
            "opened_for_write_by_v15": False,
        },
    }


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


def prep_handler(args: argparse.Namespace) -> dict[str, Any]:
    return write_static_evidence(getattr(args, "output", None) or STATIC)


def verify_bindings_handler(args: argparse.Namespace) -> dict[str, Any]:
    return verify_bindings(require_parent=Path(PARENT_PATH).exists())


def authorize_stage_handler(args: argparse.Namespace) -> dict[str, Any]:
    return authorize_stage(
        stage=args.stage,
        output=args.output,
        lead_authorized=bool(getattr(args, "lead_authorized", False)),
        acknowledges_post_hoc_rule_revision=bool(
            getattr(args, "acknowledge_rule_revision", False)
        ),
        acknowledges_inherited_vetoes=bool(getattr(args, "acknowledge_vetoes", False)),
        input_checkpoint=getattr(args, "input_checkpoint", None),
        sealed_chain_verified=bool(getattr(args, "sealed_chain_verified", False)),
    )


#: sentinel distinguishing "caller said nothing" from "caller said None".  An
#: explicit ``backend_factory=None`` keeps the implementation-only refusal.
PRODUCTION_FACTORY = "production"


def smoke_handler(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[..., Any] | None | str = PRODUCTION_FACTORY,
    smoke_runner: Callable[..., Mapping[str, Any]] = run_v15_smoke,
) -> Mapping[str, Any]:
    """Smoke.  Refuses without a fresh lead authorization for this exact stage."""
    verify_bindings(require_parent=Path(PARENT_PATH).exists())
    authorization = load_stage_authorization(args.authorization, "smoke")
    if backend_factory is None:
        raise StageAuthorizationRequired(
            "smoke requires an explicit backend factory; an implementation-only "
            "caller passing factory=None starts no training"
        )
    factory = (
        build_v15_smoke_bundle if backend_factory is PRODUCTION_FACTORY else backend_factory
    )
    bundle = factory(authorization)
    return smoke_runner(**bundle)


def _stage_refusal(stage: str) -> Callable[[argparse.Namespace], Mapping[str, Any]]:
    def handler(args: argparse.Namespace) -> Mapping[str, Any]:
        verify_bindings(require_parent=Path(PARENT_PATH).exists())
        authorization = load_stage_authorization(
            getattr(args, "authorization", ""), stage
        )
        raise StageAuthorizationRequired(
            f"{stage} is not wired in this implementation-only freeze; "
            f"authorization {authorization['authorization_sha256']} is recorded "
            "but no run may start"
        )

    handler.__name__ = f"{stage.replace('-', '_')}_handler"
    return handler


def validation_handler(
    *,
    terminal_path: str | Path,
    authorization: Mapping[str, Any],
    effective: Mapping[str, str],
    evaluate: Callable[[], tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    validate_stage_authorization(authorization, "validation-once")
    status, gates = evaluate()
    return terminal_v15(
        "validation-once",
        status,
        build_validation_terminal(
            path=Path(terminal_path),
            status=status,
            authorization=authorization,
            gates=gates,
            terminal_extra={"effective": dict(effective), "candidate": CANDIDATE},
        ),
    )


def test_handler(
    *,
    terminal_path: str | Path,
    authorization: Mapping[str, Any],
    effective: Mapping[str, str],
    evaluate: Callable[[], tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    validate_stage_authorization(authorization, "test-once")
    status, gates = evaluate()
    return terminal_v15(
        "test-once",
        status,
        build_test_terminal(
            path=Path(terminal_path),
            status=status,
            authorization=authorization,
            gates=gates,
            terminal_extra={"effective": dict(effective), "candidate": CANDIDATE},
        ),
    )


HANDLERS: dict[str, Callable[..., Any]] = {
    "prep": prep_handler,
    "verify-bindings": verify_bindings_handler,
    "authorize-stage": authorize_stage_handler,
    "smoke": smoke_handler,
    "pilot": _stage_refusal("pilot"),
    "scale-1": _stage_refusal("scale-1"),
    "scale-4": _stage_refusal("scale-4"),
    "scale-decide": _stage_refusal("scale-decide"),
    "long": _stage_refusal("long"),
    "final-train-confirm": _stage_refusal("final-train-confirm"),
    "validation-once": _stage_refusal("validation-once"),
    "test-once": _stage_refusal("test-once"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prep")
    prep.add_argument("--output", default=str(STATIC))
    sub.add_parser("verify-bindings")
    authorize = sub.add_parser("authorize-stage")
    authorize.add_argument("--stage", required=True, choices=STAGE_MODES)
    authorize.add_argument("--output", required=True)
    authorize.add_argument("--input-checkpoint")
    authorize.add_argument("--lead-authorized", dest="lead_authorized", action="store_true")
    authorize.add_argument(
        "--acknowledge-rule-revision", dest="acknowledge_rule_revision", action="store_true"
    )
    authorize.add_argument(
        "--acknowledge-vetoes", dest="acknowledge_vetoes", action="store_true"
    )
    authorize.add_argument(
        "--sealed-chain-verified", dest="sealed_chain_verified", action="store_true"
    )
    for mode in STAGE_MODES:
        stage = sub.add_parser(mode)
        stage.add_argument("--authorization", required=True)
        stage.add_argument("--physical-gpu-index", type=int, default=0)
        stage.add_argument("--device", default="cuda:0")
        stage.add_argument("--config", default=str(CONFIG.relative_to(ROOT)))
        stage.add_argument("--preregistration", default=str(PREREG.relative_to(ROOT)))
        if mode == "long":
            stage.add_argument("--world-size", type=int, default=1, choices=(1, 4))
            stage.add_argument("--input-checkpoint")
            stage.add_argument("--scale-decision")
            stage.add_argument("--resume", action="store_true")
    return parser


def command_argv(command: str) -> list[str]:
    tokens = shlex.split(command)
    index = next(
        i for i, token in enumerate(tokens) if token.endswith("scripts/train_r16_dscp_v15.py")
    )
    return tokens[index + 1 :]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = HANDLERS[args.mode](args)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
