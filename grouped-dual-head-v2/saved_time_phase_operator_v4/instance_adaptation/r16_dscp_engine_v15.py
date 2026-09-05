"""V15 engine: mask-confined correction, per-record smoke accounting, real VRAM.

This engine keeps every v14 execution and sealed-chain contract by subclassing
:class:`~.r16_dscp_engine_v14.V14ProductionBackend`, and changes exactly four
things plus the correction itself:

* the materialized correction is confined by the PARENT-energy keep mask, so it
  is exactly zero on dropped frames and the output there is bit-identical to
  the parent (fix 1, the candidate itself);
* smoke loss reduction is accounted per record through
  :class:`~.r16_dscp_training_v3.PerRecordLossLedger`; the v14 cross-record
  ``losses[0]`` vs ``losses[-1]`` reduction is not expressible here (fix 2);
* the oracle upper bound is recomputed on the record being scored; the three
  v14 hardcoded oracle constants are void and appear nowhere in this module
  (fix 3);
* ``normalized_coefficient_energy`` is the dimensionless v15 definition
  (fix 4, in :mod:`.r16_dscp_training_v3`);
* peak VRAM is sampled on every cached path and the VRAM gate refuses to pass
  on an unmeasured or zero peak (fix 5).

Standing statements, kept in code so they travel with the artifacts:

* the correction is an output-field data fit plus a feature-driven ansatz; it is
  not a physics or PDE residual and not deployment-time truth supervision, and
  no PDE residual is evaluated anywhere in this pipeline;
* nothing produced by the smoke stage is evidence of learning or convergence;
* smoke evidence may never promote a candidate to pilot or long.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from .r16_dscp import c1_causal_mask
from .r16_dscp_engine_v3 import canonical_sha
from .r16_dscp_engine_v4 import CandidateSpool, V4Prepared
from .r16_dscp_engine_v7 import validate_checkpoint_identity
from .r16_dscp_engine_v8 import ROLLBACK, terminal_base
from .r16_dscp_engine_v14 import V14ProductionBackend
from .r16_dscp_training_v2 import BindingRefusal
from .r16_dscp_training_v3 import (
    ACCEPTANCE_CONVENTION,
    CudaPeakVram,
    EPS_SQUARED,
    PerRecordLossLedger,
    TAU,
    TRAINING_PROXY_CONVENTION,
    TRAIN_VRAM_LIMIT_BYTES,
    apply_confined_correction,
    coefficient_energy_definition,
    confined_oracle_upper_bound,
    full_time_keep_mask,
    loss_specification,
    mask_provenance,
    masked_confined_loss,
    vram_gate,
)

CANDIDATE = "r16_dscp_v15"

#: smoke budget, transcribed from the frozen v15 preregistration
SMOKE_MAX_UPDATES = 192
SMOKE_TOTAL_S = 600.0
SMOKE_TRAINING_DEADLINE_S = 420.0
SMOKE_FINAL_RESERVE_S = 180.0
SMOKE_QUICK_INTERVAL = 32

#: inherited v14 gate thresholds; not moved toward any measured value
LOSS_REDUCTION_MIN = 0.80
ORACLE_GAIN_FRACTION = 0.5
EARLY_STOP_FRACTION = 0.2

#: the v14 oracle constants are void.  Their numeric values are deliberately
#: absent from this module: they were measured on train_uniform_00102,
#: train_layered_01032 and a SYNTHETIC marmousi record, then used to score
#: train_uniform_00321 / train_layered_00564 / train_marmousi_00385.  A
#: same-convention recomputation on the scored records disagreed, worst case by
#: about 2.5x on marmousi.  v15 recomputes the bound on the scored record.
VOIDED_ORACLE_PROVENANCE = {
    "status": "void",
    "v14_source_records": ("train_uniform_00102", "train_layered_01032", "synthetic_marmousi"),
    "v14_scored_records": ("train_uniform_00321", "train_layered_00564", "train_marmousi_00385"),
    "defects": (
        "constants were transplanted across records",
        "the marmousi constant came from a synthetic sample",
        "no constant was recomputed on the record it scored",
    ),
    "v15_replacement": "confined_oracle_upper_bound recomputed on the scored record",
    "numeric_values_present_in_v15_code": False,
}

SMOKE_DISCLAIMERS = {
    "is_learning_or_convergence_evidence": False,
    "may_promote_to_pilot_or_long": False,
    "loss_gate_role": "wiring sanity check on each record separately",
    "oracle_role": "train-only upper bound, never an achieved result",
    "acceptance_convention": ACCEPTANCE_CONVENTION,
    "training_proxy_convention_not_for_acceptance": TRAINING_PROXY_CONVENTION,
    "pde_residual_used": False,
    "deployment_time_truth_supervision": False,
}


class V15GateRefusal(BindingRefusal):
    """A v15 gate was asked to pass on evidence it does not have."""


class VramMeasurementRefusal(V15GateRefusal):
    """Peak VRAM was never measured, so the VRAM gate cannot be evaluated."""


class OracleProvenanceRefusal(V15GateRefusal):
    """An oracle bound was not recomputed on the record it is scoring."""


class SmokeAccountingRefusal(V15GateRefusal):
    """Smoke loss accounting was requested across records."""


class SmokePromotionRefusal(V15GateRefusal):
    """Smoke evidence was offered as a promotion to pilot or long."""


def refuse_promotion_from_smoke(target_stage: str) -> None:
    """Smoke evidence can never promote.  Inherited veto (f)."""
    raise SmokePromotionRefusal(
        f"smoke evidence must not promote to {target_stage}; a fresh lead "
        "authorization is required for every stage"
    )


class V15ProductionBackend(V14ProductionBackend):
    """V14 contracts, mask-confined correction, and real peak VRAM accounting."""

    def __init__(self, *args: Any, tau: float = TAU, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.spool = CandidateSpool(
            self.run_dir,
            run_digest=self.lineage.run_digest,
            rank=int(os.environ.get("RANK", "0")),
            owned_root="/dev/shm/r16_dscp_v15",
        )
        self.tau = float(tau)
        self.vram = CudaPeakVram(self.device).reset()
        self.mask_events: list[dict[str, Any]] = []
        self._sample_vram()

    # -- fix 5: sample the real peak on every path the run actually takes ----

    def _sample_vram(self) -> int:
        peak = self.vram.sample()
        self.peak_bytes = max(int(self.peak_bytes), int(peak))
        return int(self.peak_bytes)

    def preload(self, index: int) -> Any:
        entry = super().preload(index)
        self._sample_vram()
        return entry

    def cached_prepare(self, index: int, key_suffix: str = "cached") -> V4Prepared:
        prepared = super().cached_prepare(index, key_suffix)
        self._sample_vram()
        return prepared

    def update_cached(self, index: int) -> Mapping[str, Any]:
        row = super().update_cached(index)
        self._sample_vram()
        return row

    def measure_cached(self, index: int) -> Mapping[str, Any]:
        row = super().measure_cached(index)
        self._sample_vram()
        return row

    def backward_cached(self, index: int, scale: float) -> Mapping[str, Any]:
        row = super().backward_cached(index, scale)
        self._sample_vram()
        return row

    def score_cached(self, index: int, release: bool = False) -> Mapping[str, Any]:
        row = super().score_cached(index, release)
        self._sample_vram()
        return row

    def update(self, prepared: V4Prepared) -> Mapping[str, Any]:
        row = super().update(prepared)
        self._sample_vram()
        return row

    def score(self, prepared: V4Prepared, truth: torch.Tensor, cleanup: bool = True) -> Mapping[str, Any]:
        row = super().score(prepared, truth, cleanup)
        self._sample_vram()
        return row

    def vram_payload(self) -> dict[str, Any]:
        self._sample_vram()
        payload = self.vram.payload()
        payload["backend_peak_bytes"] = int(self.peak_bytes)
        payload["sampled_paths"] = [
            "preload",
            "cached_prepare",
            "update_cached",
            "measure_cached",
            "backward_cached",
            "score_cached",
            "forward_loss",
            "resources",
        ]
        payload["v14_defect"] = (
            "peak_vram_bytes was 0 because only V4ProductionBackend.prepare "
            "sampled CUDA memory and no cached path did"
        )
        return payload

    def resources(self, saved: Mapping[str, Any], world_size: int = 1) -> Mapping[str, Any]:
        self._sample_vram()
        result = dict(super().resources(saved, world_size))
        payload = self.vram_payload()
        result["peak_bytes"] = max(int(result.get("peak_bytes", 0)), int(self.vram.peak_bytes))
        result["peak_vram_bytes"] = int(result["peak_bytes"])
        result["vram"] = payload
        return result

    # -- fix 1: mask-confined correction -----------------------------------

    def keep_mask(self, parent: torch.Tensor, k1: int) -> tuple[torch.Tensor, float]:
        """Parent-energy keep mask for one record.  Truth is not an input."""
        field = torch.as_tensor(parent)
        field = field[0] if field.ndim == 4 else field
        return full_time_keep_mask(field.detach().float(), k1=int(k1), tau=self.tau)

    def _materialize(
        self, parent: torch.Tensor, coefficient: torch.Tensor, route_index: int, k1: int
    ) -> torch.Tensor:
        corrected = parent.float().clone()
        if route_index >= 0:
            basis = self.candidate.bases[route_index].float()
            correction = torch.einsum("tr,rhw->thw", basis, coefficient[0])
            ramp = c1_causal_mask(
                parent.shape[1], int(k1), device=parent.device, dtype=torch.float32
            )
            correction = correction * ramp[:, None, None]
            correction = torch.cat(
                (torch.zeros_like(correction[:, :1]), correction[:, 1:]), dim=1
            )
            keep, floor = self.keep_mask(parent, int(k1))
            keep = keep.to(correction.device)
            confined_field = apply_confined_correction(
                parent[0].float(), correction, keep
            )
            corrected = torch.cat((confined_field[None], corrected[1:]), dim=0)
            self.mask_events.append(
                {
                    "k1": int(k1),
                    "route_index": int(route_index),
                    "kept_frames": int(keep.sum().item()),
                    "dropped_frames": int((~keep).sum().item()),
                    "parent_energy_floor": float(floor),
                    "tau": float(self.tau),
                    "confined": True,
                }
            )
        self._sample_vram()
        return corrected

    def _forward_loss(self, prepared: V4Prepared, truth: torch.Tensor):
        features = prepared.features.to(self.device).float()
        abstain = prepared.route_index < 0
        coefficient = self._coefficients(
            features, prepared.route_index, abstain, prepared.condition
        )
        parent = prepared.args[7]
        k1 = int(prepared.public.observed_indices[1])
        adapted = self._materialize(parent, coefficient, prepared.route_index, k1)
        keep, _floor = self.keep_mask(parent, k1)
        losses = masked_confined_loss(
            adapted[:, k1 + 1 :].float(),
            torch.as_tensor(truth).to(self.device).float(),
            coefficient,
            parent[:, k1 + 1 :].float(),
            keep[k1 + 1 :].to(self.device),
            tau=self.tau,
        )
        self._sample_vram()
        return losses, coefficient

    def mask_ledger(self) -> dict[str, Any]:
        return {
            **mask_provenance(tau=self.tau),
            "materializations_confined": len(self.mask_events),
            "unconfined_materializations": 0,
            "events": list(self.mask_events[-8:]),
        }


def complete_identity_v15(**values: Any) -> dict[str, Any]:
    payload = {
        "schema": "r16_dscp_v15_checkpoint_identity_v1",
        "candidate": CANDIDATE,
        **values,
    }
    payload["identity_digest"] = canonical_sha(payload)
    validate_checkpoint_identity(payload)
    return payload


def terminal_v15(mode: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(payload),
        "schema": f"r16_dscp_v15_{mode.replace('-', '_')}_terminal_v1",
        "candidate": CANDIDATE,
        "mode": mode,
        "status": status,
        "decision": status,
        "rollback": ROLLBACK,
    }


# ---------------------------------------------------------------------------
# fix 3: per-record oracle, recomputed on the record being scored
# ---------------------------------------------------------------------------


def record_oracle_bound(
    *,
    sample_id: str,
    family: str,
    parent_future: torch.Tensor,
    truth_future: torch.Tensor,
    basis_future: torch.Tensor,
    tau: float = TAU,
    spatial_chunk: int = 4096,
) -> dict[str, Any]:
    """Recompute the confined oracle bound for one record and tag its identity."""
    payload = dict(
        confined_oracle_upper_bound(
            parent_future, truth_future, basis_future, tau=tau, spatial_chunk=spatial_chunk
        )
    )
    payload["sample_id"] = str(sample_id)
    payload["family"] = str(family)
    payload["voided_v14_constants"] = dict(VOIDED_ORACLE_PROVENANCE)
    return payload


def validate_oracle_provenance(sample_id: str, payload: Mapping[str, Any]) -> None:
    if str(payload.get("sample_id")) != str(sample_id):
        raise OracleProvenanceRefusal(
            f"oracle bound for {payload.get('sample_id')} cannot score {sample_id}"
        )
    if not payload.get("recomputed_on_the_scored_record"):
        raise OracleProvenanceRefusal("oracle bound was not recomputed on the scored record")
    if payload.get("transplanted_constants_used") or payload.get("synthetic_records_used"):
        raise OracleProvenanceRefusal("oracle bound carries transplanted or synthetic provenance")


def acceptance_convention_gain(row: Mapping[str, Any]) -> float:
    """Per-record convention (iii) gain from a score row.

    ``aggregate_rel_l2`` and ``parent_rel_l2`` in a score row are exactly the
    global energy relative L2 of candidate and parent over the future window.
    """
    parent = float(row["parent_rel_l2"])
    candidate = float(row["aggregate_rel_l2"])
    return (parent - candidate) / max(abs(parent), EPS_SQUARED)


def oracle_gain_gate(
    score_records: Sequence[Mapping[str, Any]],
    oracle_by_record: Mapping[str, Mapping[str, Any]],
    *,
    fraction: float = ORACLE_GAIN_FRACTION,
) -> dict[str, Any]:
    """Per-record oracle-fraction gate on the acceptance convention.

    Each record is compared against the bound recomputed on itself.  A missing
    bound is a refusal, not a pass, and a nonpositive bound cannot be cleared.
    """
    rows: dict[str, Any] = {}
    for row in score_records:
        sample_id = str(row["sample_id"])
        payload = oracle_by_record.get(sample_id)
        if payload is None:
            raise OracleProvenanceRefusal(f"no oracle bound recomputed for {sample_id}")
        validate_oracle_provenance(sample_id, payload)
        bound = float(payload["acceptance_convention_gain"])
        achieved = acceptance_convention_gain(row)
        required = float(fraction) * bound
        rows[sample_id] = {
            "sample_id": sample_id,
            "family": str(row.get("family", payload.get("family", ""))),
            "convention": ACCEPTANCE_CONVENTION,
            "achieved_gain": achieved,
            "oracle_upper_bound_gain": bound,
            "required_gain": required,
            "oracle_bound_positive": bound > 0.0,
            "passed": bound > 0.0 and achieved >= required,
        }
    return {
        "fraction": float(fraction),
        "per_record": rows,
        "recomputed_per_record": True,
        "transplanted_constants_used": False,
        "voided_v14_constants": dict(VOIDED_ORACLE_PROVENANCE),
        "failing_records": sorted(k for k, v in rows.items() if not v["passed"]),
        "passed": bool(rows) and all(v["passed"] for v in rows.values()),
    }


# ---------------------------------------------------------------------------
# fix 2 and fix 5: the v15 smoke gates
# ---------------------------------------------------------------------------


def v14_cross_record_loss_reduction(losses: Sequence[float]) -> float:
    """The v14 formula, reproduced only so tests can show what it does.

    ``(losses[0] - losses[-1]) / max(|losses[0]|, 1e-30)`` over a round-robin
    stream.  With three records and a multiple-of-three update count the first
    and last observations are different records, so this number is a record
    ordering artifact.  It must never be used as a gate.
    """
    if len(losses) < 2:
        raise SmokeAccountingRefusal("cross-record reduction needs two observations")
    initial = float(losses[0])
    final = float(losses[-1])
    return (initial - final) / max(abs(initial), EPS_SQUARED)


def smoke_gates_v15(
    *,
    ledger: PerRecordLossLedger,
    score_records: Sequence[Mapping[str, Any]],
    resources: Mapping[str, Any],
    oracle_by_record: Mapping[str, Mapping[str, Any]],
    loss_reduction_min: float = LOSS_REDUCTION_MIN,
    oracle_fraction: float = ORACLE_GAIN_FRACTION,
    vram_limit_bytes: int = TRAIN_VRAM_LIMIT_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Smoke gates with per-record loss accounting and a measured VRAM gate.

    Thresholds are the inherited v14 values.  None of them is moved toward a
    measured number.  Passing these gates is not evidence of learning and does
    not authorize pilot or long.
    """
    if not isinstance(ledger, PerRecordLossLedger):
        raise SmokeAccountingRefusal("smoke loss accounting requires a per-record ledger")
    vram_payload = dict(resources.get("vram") or {})
    if not vram_payload:
        raise VramMeasurementRefusal("resources carry no vram measurement payload")
    loss = ledger.gate(threshold=float(loss_reduction_min))
    oracle = oracle_gain_gate(score_records, oracle_by_record, fraction=oracle_fraction)
    vram = vram_gate(vram_payload, limit_bytes=int(vram_limit_bytes))
    nonworse = [bool(row.get("nonworse")) for row in score_records]
    finite = all(
        all(
            isinstance(row.get(key), float) and row[key] == row[key]
            for key in ("aggregate_rel_l2", "parent_rel_l2")
        )
        for row in score_records
    )
    gates = {
        "loss": loss,
        "oracle_gain": oracle,
        "nonworse": {
            "value": int(sum(nonworse)),
            "threshold": len(nonworse),
            "passed": bool(nonworse) and all(nonworse),
        },
        "finite": {"value": finite, "passed": finite},
        "vram": vram,
        "space": {
            "value": bool(resources.get("space_passed")),
            "passed": bool(resources.get("space_passed")),
        },
    }
    metrics = {
        "per_record_loss": ledger.per_record(),
        "update_count": ledger.update_count,
        "records": [dict(row) for row in score_records],
        "oracle_by_record": {k: dict(v) for k, v in oracle_by_record.items()},
        "vram": vram_payload,
        "mask": mask_provenance(),
        "loss_specification": loss_specification(),
        "coefficient_energy": coefficient_energy_definition(),
        "disclaimers": dict(SMOKE_DISCLAIMERS),
        "voided_v14_oracle_constants": dict(VOIDED_ORACLE_PROVENANCE),
    }
    return gates, metrics


def run_v15_smoke(
    *,
    backend: Any,
    records: Sequence[int],
    quick_gate: Callable[[Sequence[Mapping[str, Any]]], bool],
    checkpoint: Callable[[], Mapping[str, Any]],
    resource_snapshot: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    terminal: Callable[[Mapping[str, Any]], None],
    oracle: Callable[[int, Mapping[str, Any]], Mapping[str, Any]],
    lineage: Mapping[str, Any],
    effective: Mapping[str, Any],
    parent: Mapping[str, Any],
    record_key: Callable[[int], str] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Mapping[str, Any]:
    """Round-robin smoke with per-record loss accounting.

    The flat ordered loss list of v14 is never formed.  Early stop requires
    every record to have fallen against its OWN first observation, and the exit
    gate is judged per record.
    """
    key_of = record_key or (lambda index: f"record_{int(index)}")
    started = clock()
    for index in records:
        backend.preload(index)
    deadline = min(
        started + SMOKE_TRAINING_DEADLINE_S,
        started + SMOKE_TOTAL_S - SMOKE_FINAL_RESERVE_S,
    )
    ledger = PerRecordLossLedger()
    updates = 0
    early = False
    while updates < SMOKE_MAX_UPDATES and clock() < deadline:
        index = records[updates % len(records)]
        ledger.observe(key_of(index), float(backend.update_cached(index)["loss"]))
        updates += 1
        if updates % SMOKE_QUICK_INTERVAL == 0:
            quick = [backend.score_cached(i) for i in records]
            rows = ledger.per_record()
            fallen = bool(rows) and all(
                row["sufficient_observations"]
                and row["final_loss"] <= EARLY_STOP_FRACTION * abs(row["initial_loss"])
                for row in rows.values()
            )
            if fallen and quick_gate(quick):
                early = True
                break
    cache = backend.cache.payload()
    latency = backend.protocol.payload()
    if started + SMOKE_TOTAL_S - clock() < SMOKE_FINAL_RESERVE_S:
        payload = terminal_v15(
            "smoke",
            "fail_budget",
            terminal_base(
                mode="smoke",
                status="fail_budget",
                lineage=lineage,
                effective=effective,
                parent=parent,
                checkpoint=None,
                updates=updates,
                early_stop=early,
                access_ledger_digest=None,
                cache=cache,
                latency=latency,
                vram=backend.vram_payload(),
                peak_vram_bytes=int(backend.vram.peak_bytes),
                mask=backend.mask_ledger(),
                per_record_loss=ledger.per_record(),
                disclaimers=dict(SMOKE_DISCLAIMERS),
                gates={"budget": {"passed": False}},
            ),
        )
        terminal(payload)
        backend.cache.clear()
        return payload
    scores = [dict(backend.score_cached(i)) for i in records]
    oracle_by_record = {}
    for index, row in zip(records, scores):
        payload = dict(oracle(index, row))
        validate_oracle_provenance(str(row["sample_id"]), payload)
        oracle_by_record[str(row["sample_id"])] = payload
    saved = dict(checkpoint())
    resources = dict(resource_snapshot(saved))
    resources["wall_s"] = clock() - started
    gates, metrics = smoke_gates_v15(
        ledger=ledger,
        score_records=scores,
        resources=resources,
        oracle_by_record=oracle_by_record,
    )
    total = clock() - started
    status = (
        "passed"
        if all(gate["passed"] for gate in gates.values()) and total <= SMOKE_TOTAL_S
        else "fail_budget"
        if total > SMOKE_TOTAL_S
        else "fail_gate"
    )
    payload = terminal_v15(
        "smoke",
        status,
        terminal_base(
            mode="smoke",
            status=status,
            lineage=lineage,
            effective=effective,
            parent=parent,
            checkpoint=saved,
            updates=updates,
            early_stop=early,
            total_s=total,
            access_ledger_digest=canonical_sha([row["ledger_digest"] for row in scores]),
            cache=cache,
            latency=latency,
            vram=metrics["vram"],
            peak_vram_bytes=int(backend.vram.peak_bytes),
            mask=backend.mask_ledger(),
            per_record_loss=metrics["per_record_loss"],
            disclaimers=dict(SMOKE_DISCLAIMERS),
            gates=gates,
            metrics=metrics,
        ),
    )
    terminal(payload)
    backend.cache.clear()
    return payload


__all__ = [
    "CANDIDATE",
    "EARLY_STOP_FRACTION",
    "LOSS_REDUCTION_MIN",
    "ORACLE_GAIN_FRACTION",
    "OracleProvenanceRefusal",
    "SMOKE_DISCLAIMERS",
    "SMOKE_MAX_UPDATES",
    "SmokeAccountingRefusal",
    "SmokePromotionRefusal",
    "V15GateRefusal",
    "V15ProductionBackend",
    "VOIDED_ORACLE_PROVENANCE",
    "VramMeasurementRefusal",
    "acceptance_convention_gain",
    "complete_identity_v15",
    "oracle_gain_gate",
    "record_oracle_bound",
    "refuse_promotion_from_smoke",
    "run_v15_smoke",
    "smoke_gates_v15",
    "terminal_v15",
    "v14_cross_record_loss_reduction",
    "validate_oracle_provenance",
]
