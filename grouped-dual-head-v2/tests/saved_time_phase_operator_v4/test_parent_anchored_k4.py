from __future__ import annotations

import hashlib
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
import h5py

from scripts.audit_transfer_dg_parent_anchored_k4 import (
    EXPECTED_LEDGER,
    k4_means,
    mechanism_ledger_passed,
    mechanism_classification,
    partial_truth_ledger,
    sampling_cv,
    validate_mechanism_contract,
)
from scripts.launch_transfer_dg_parent_anchored_k4 import (
    validate_launcher_contract,
    validate_prerequisites,
    verify_child_launch,
)
from scripts.train_transfer_dg_parent_anchored_k4 import (
    EXPECTED_LR_UPDATE_204,
    EXPECTED_ROLE_DIGESTS,
    OFFSETS,
    SCHEDULER_T_MAX,
    STARTS,
    TruthAccessGuard,
    audit_role_metadata,
    checkpoint_purpose_allows_pilot_initialization,
    cross_role_overlap_counts,
    expected_lr,
    fresh_model_cpu,
    k4_accumulate_and_step,
    k4_indices,
    pilot_acceptance,
    sha256,
    training_failure_payload,
    validate_training_invocation,
)


ROOT = Path(__file__).resolve().parents[2]


def test_all_anchors_have_four_distinct_windows_and_balanced_frequency() -> None:
    counts = np.zeros(len(STARTS), dtype=np.int64)
    assert OFFSETS == (0, 3, 6, 9)
    for anchor in range(len(STARTS)):
        indices = k4_indices(anchor)
        assert len(indices) == len(set(indices)) == 4
        for index in indices:
            counts[index] += 1
    assert counts.tolist() == [4] * 13


def test_synthetic_k4_mean_algebra_cv_and_linear_median() -> None:
    base = torch.tensor([2.0, -1.0], dtype=torch.float64)
    perturbations = [
        torch.tensor([math.cos(2 * math.pi * index / 13), math.sin(2 * math.pi * index / 13)])
        for index in range(13)
    ]
    k1 = [base + perturbation for perturbation in perturbations]
    ght = torch.stack(k1).mean(0)
    k4 = k4_means(k1)
    assert torch.allclose(torch.stack(k4).mean(0), ght, rtol=0.0, atol=1.0e-15)
    assert sampling_cv(k4, ght) < sampling_cv(k1, ght)
    shifted_center = ght + torch.tensor([0.25, -0.5], dtype=torch.float64)
    assert sampling_cv(k4, torch.stack(k4).mean(0)) != pytest.approx(
        sampling_cv(k4, shifted_center)
    )
    values = [0.9, 0.1, 0.7, 0.3]
    assert float(np.median(values)) == pytest.approx(0.5)


def _mechanism_metrics() -> dict:
    return {
        "records": [
            {
                "k4_mean_vs_ht_cosine": 0.999999,
                "k4_mean_vs_ht_relative_norm_difference": 1.0e-5,
                "cv_k4": 1.0,
            }
            for _ in range(4)
        ],
        "median_cv_k4_over_k1": 0.70,
        "k4_vs_own_full_main_cosine_median": 0.95,
        "aggregate_mean_k4_vs_full_main_cosine": 0.995,
        "k4_clip_fraction": 0.249999,
        "finite": True,
        "rollback": True,
        "truth_ledger_passed": True,
    }


def test_mechanism_classification_exact_boundaries() -> None:
    metrics = _mechanism_metrics()
    assert mechanism_classification(metrics)["status"] == "passed"
    metrics = _mechanism_metrics()
    metrics["k4_clip_fraction"] = 0.25
    assert mechanism_classification(metrics)["status"] == "rejected"
    metrics = _mechanism_metrics()
    metrics["records"][0]["k4_mean_vs_ht_cosine"] = 0.999998999
    assert mechanism_classification(metrics)["status"] == "rejected"
    metrics = _mechanism_metrics()
    metrics["records"][0]["cv_k4"] = 1.000001
    metrics["records"][1]["cv_k4"] = 1.000001
    assert mechanism_classification(metrics)["status"] == "rejected"


class _CountingOptimizer:
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.parameter = parameter
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.zero_calls += 1
        self.parameter.grad = None

    def step(self) -> None:
        self.step_calls += 1


class _CountingScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


def test_loss_quarter_accumulation_and_single_clip_step_scheduler() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.5], dtype=torch.float64))
    targets = [0.0, 1.0, 2.0, 4.0]
    individual = [2.0 * (float(parameter.detach()) - target) for target in targets]
    optimizer = _CountingOptimizer(parameter)
    scheduler = _CountingScheduler()
    clip_calls: list[float] = []

    def loss_factory(index: int) -> torch.Tensor:
        return (parameter - targets[index]).square().sum()

    def clipper(parameters, threshold):
        clip_calls.append(threshold)
        return torch.linalg.vector_norm(torch.cat([value.grad.flatten() for value in parameters]))

    result = k4_accumulate_and_step(
        loss_factory=loss_factory,
        optimizer=optimizer,
        scheduler=scheduler,
        parameters=(parameter,),
        clipper=clipper,
    )
    assert float(parameter.grad) == pytest.approx(float(np.mean(individual)))
    assert optimizer.zero_calls == optimizer.step_calls == scheduler.step_calls == 1
    assert clip_calls == [1.0]
    assert math.isfinite(result["mean_loss"])


def test_cosine_scheduler_tmax_and_update_204_lr() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=3.0e-4, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX, eta_min=3.0e-6
    )
    for _ in range(204):
        optimizer.step()
        scheduler.step()
    assert scheduler.last_epoch == 204
    assert scheduler.get_last_lr()[0] == pytest.approx(EXPECTED_LR_UPDATE_204, rel=1.0e-12, abs=1.0e-15)
    assert expected_lr(204) == pytest.approx(EXPECTED_LR_UPDATE_204, rel=1.0e-15)


def test_truth_guard_purpose_role_rejections_and_ledger() -> None:
    guard = TruthAccessGuard(mechanism_panel={"fit-a"})
    guard.authorize(
        purpose="mechanism",
        role="fit",
        sample_id="fit-a",
        group_id="group-a",
        sample_sha256="hash-a",
        start=2,
        stop=66,
    )
    guard.authorize(
        purpose="calibration",
        role="calibration",
        sample_id="cal-a",
        group_id="group-b",
        sample_sha256="hash-b",
        start=0,
        stop=401,
    )
    with pytest.raises(PermissionError):
        guard.authorize(
            purpose="training",
            role="calibration",
            sample_id="cal-a",
            group_id="group-b",
            sample_sha256="hash-b",
            start=2,
            stop=10,
        )
    with pytest.raises(PermissionError):
        guard.authorize(
            purpose="calibration",
            role="confirmation",
            sample_id="confirm-a",
            group_id="group-c",
            sample_sha256="hash-c",
            start=0,
            stop=401,
        )
    with pytest.raises(PermissionError):
        guard.authorize(
            purpose="mechanism",
            role="fit",
            sample_id="fit-not-panel",
            group_id="group-d",
            sample_sha256="hash-d",
            start=2,
            stop=3,
        )
    ledger = guard.summary()
    assert ledger["authorized_call_count"] == 2
    assert ledger["purpose_call_counts"] == {"calibration": 1, "mechanism": 1}
    assert ledger["role_call_counts"] == {"calibration": 1, "fit": 1}
    assert ledger["slice_start_min"] == 0
    assert ledger["slice_stop_max_exclusive"] == 401
    assert ledger["confirmation_opened"] is False


def test_three_pair_role_overlap_accounting() -> None:
    role_sets = {
        "fit": {"sample_ids": {"f"}, "groups": {"gf"}, "sample_hashes": {"hf"}},
        "calibration": {"sample_ids": {"c"}, "groups": {"gc"}, "sample_hashes": {"hc"}},
        "confirmation": {"sample_ids": {"q"}, "groups": {"gq"}, "sample_hashes": {"hq"}},
    }
    assert set(cross_role_overlap_counts(role_sets)) == {
        f"{left}_{right}_{field}"
        for left, right in (
            ("fit", "calibration"),
            ("fit", "confirmation"),
            ("calibration", "confirmation"),
        )
        for field in ("sample_ids", "groups", "sample_hashes")
    }
    assert not any(cross_role_overlap_counts(role_sets).values())
    role_sets["confirmation"]["groups"].add("gf")
    assert cross_role_overlap_counts(role_sets)["fit_confirmation_groups"] == 1


def test_mechanism_ledger_exact_contract() -> None:
    samples = [
        ("train_uniform_00014", "train:uniform:00014", "c525b99d2fe42f5732d61a35c6ffdff1a616d8c06f5c780355a81f25645ba970"),
        ("train_layered_00696", "train:layered:00174", "14693030b443d1c837abdb448b0ddd810fb3199f30b5e2eb4f15d8dd26ce9ba9"),
        ("train_anomaly_00096", "train:anomaly:00048", "6a876f6e8d1287a64a2feeca9e0a00015c884c0ed31f4124350f7027a34e840d"),
        ("train_marmousi_00230", "train:marmousi:x1300.0:z100.0", "dfeb947dcc5e2cce0a55596a651062b1da8973e5f395fdd1ea0fd3fb849d799a"),
    ]
    guard = TruthAccessGuard(mechanism_panel={sample[0] for sample in samples})
    for sample_id, group_id, sample_hash in samples:
        slices = [(2, 3)] * 25 + [(2, 66)] * 7 + [(0, 401)]
        for start, stop in slices:
            guard.authorize(
                purpose="mechanism",
                role="fit",
                sample_id=sample_id,
                group_id=group_id,
                sample_sha256=sample_hash,
                start=start,
                stop=stop,
            )
    ledger = guard.summary()
    assert ledger == EXPECTED_LEDGER
    assert mechanism_ledger_passed(ledger)
    ledger["authorized_call_count"] = 131
    assert not mechanism_ledger_passed(ledger)


def test_real_metadata_style_role_digests_and_intra_role_uniqueness() -> None:
    cache_path = ROOT / "results/transfer_dg_parent_anchored_block32_r2_20260904/cache_full_136rec.h5"
    source_path = Path(
        "/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5"
    )
    with h5py.File(cache_path, "r", swmr=True) as cache_file, h5py.File(
        source_path, "r", swmr=True
    ) as source_file:
        roles = cache_file["role"].asstr()[:]
        sample_ids = cache_file["sample_id"].asstr()[:]
        group_ids = cache_file["group_id"].asstr()[:]
        families = cache_file["family"].asstr()[:]
        source_indices = np.asarray(cache_file["source_index"], dtype=np.int64)
        sample_hashes = np.asarray(
            [
                source_file["sample_sha256"][int(index)].decode()
                if isinstance(source_file["sample_sha256"][int(index)], bytes)
                else str(source_file["sample_sha256"][int(index)])
                for index in source_indices
            ],
            dtype=object,
        )

        class MetadataFixture:
            def __init__(self) -> None:
                self.roles = roles
                self.sample_ids = sample_ids
                self.group_ids = group_ids
                self.families = families
                self.source_indices = source_indices
                self.sample_hashes = sample_hashes

            def positions(self, role: str) -> list[int]:
                return np.flatnonzero(self.roles == role).astype(np.int64).tolist()

        fixture = MetadataFixture()
        audit = audit_role_metadata(fixture)
    for role, expected in EXPECTED_ROLE_DIGESTS.items():
        assert audit["roles"][role]["digests"] == expected
        positions = fixture.positions(role)
        assert len({fixture.sample_ids[index] for index in positions}) == len(positions)
        assert len({fixture.sample_hashes[index] for index in positions}) == len(positions)
    assert not any(audit["overlap_counts"].values())


def test_constructor_failure_can_report_zero_call_partial_ledger() -> None:
    guard = TruthAccessGuard()
    try:
        raise RuntimeError("synthetic metadata constructor failure")
    except RuntimeError:
        ledger = partial_truth_ledger(guard)
    assert ledger["authorized_call_count"] == 0
    assert ledger["purpose_call_counts"] == {}
    assert ledger["role_call_counts"] == {}
    assert ledger["slice_start_min"] is None
    assert ledger["slice_stop_max_exclusive"] is None
    assert ledger["confirmation_opened"] is False
    assert ledger["validation_opened"] is False
    assert ledger["test_id_opened"] is False


def test_fresh_initial_and_parameter_manifest_digests_are_stable() -> None:
    first, first_state, first_manifest = fresh_model_cpu(372)
    second, second_state, second_manifest = fresh_model_cpu(372)
    assert first_state == second_state
    assert first_manifest == second_manifest
    assert first_state == "b2bfbb598fc608089acd0b7e080ba1c5ee5f4b3a87ed5dd333abd49cb9bef196"
    assert first_manifest == "f8ee8839801b3710d48d0e24ef18553ead8df785efb39e2dbd8953aab9b6c846"
    assert sum(parameter.numel() for parameter in first.parameters()) == 91396
    assert len(tuple(first.named_parameters())) == 62


def _acceptance_metrics() -> dict:
    return {
        "candidate_mean": 0.47955565810136125,
        "nonworse_count": 30,
        "per_family_candidate": {
            "uniform": 0.37022280539921215,
            "layered": 0.3149375678241338,
            "anomaly": 0.5241317602092531,
            "marmousi": 0.6795251190565753,
        },
    }


def test_pilot_acceptance_exact_boundaries() -> None:
    assert pilot_acceptance(
        _acceptance_metrics(), update=204, learning_rate=EXPECTED_LR_UPDATE_204
    )["passed"]
    metrics = _acceptance_metrics()
    metrics["candidate_mean"] += 1.0e-12
    assert not pilot_acceptance(metrics, update=204, learning_rate=EXPECTED_LR_UPDATE_204)["passed"]
    assert not pilot_acceptance(
        _acceptance_metrics(), update=203, learning_rate=EXPECTED_LR_UPDATE_204
    )["passed"]


def test_smoke_checkpoint_purpose_forbids_pilot_initialization() -> None:
    assert not checkpoint_purpose_allows_pilot_initialization(
        {"purpose": "smoke_only", "pilot_init_forbidden": True}
    )
    assert checkpoint_purpose_allows_pilot_initialization(
        {"purpose": "pilot", "pilot_init_forbidden": False}
    )


def test_launcher_prerequisites_require_passed_bound_files(tmp_path: Path) -> None:
    mechanism = tmp_path / "mechanism.json"
    smoke = tmp_path / "smoke.json"
    mechanism.write_text("{}\n", encoding="utf-8")
    smoke.write_text("{}\n", encoding="utf-8")
    prereg = {
        "prerequisites": {
            "mechanism": {"status": "passed", "path": str(mechanism), "sha256": sha256(mechanism)},
            "smoke": {"status": "passed", "path": str(smoke), "sha256": sha256(smoke)},
        }
    }
    validate_prerequisites(prereg)
    prereg["prerequisites"]["smoke"]["status"] = "pending"
    with pytest.raises(RuntimeError):
        validate_prerequisites(prereg)


def _stage_prereg(tmp_path: Path, *, status: str) -> tuple[dict, argparse.Namespace]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prereg_path = tmp_path / "prereg.json"
    cache = tmp_path / "cache.h5"
    source = tmp_path / "source.h5"
    output = tmp_path / "output"
    mechanism = tmp_path / "mechanism.json"
    smoke = tmp_path / "smoke.json"
    for path in (cache, source, mechanism, smoke):
        path.write_text("bound\n", encoding="utf-8")
    prereg = {
        "schema": "transfer_dg_parent_anchored_k4_preregistration_v1",
        "candidate": "transfer_dg_parent_anchored_k4_r4_20260904",
        "status": status,
        "paths": {
            "preregistration": str(prereg_path),
            "cache": str(cache),
            "source_h5": str(source),
            "smoke_output_dir": str(output),
            "pilot_output_dir": str(output),
        },
        "prerequisites": {
            "mechanism": {"status": "passed", "path": str(mechanism), "sha256": sha256(mechanism)},
            "smoke": {"status": "passed", "path": str(smoke), "sha256": sha256(smoke)},
        },
    }
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    args = argparse.Namespace(
        smoke=status.startswith("mechanism_passed"),
        pilot=status.startswith("pilot_pending"),
        preregistration=prereg_path,
        cache=cache,
        source_h5=source,
        output_dir=output,
    )
    return prereg, args


def test_training_status_prerequisite_and_pre_output_path_validation(tmp_path: Path) -> None:
    prereg, args = _stage_prereg(
        tmp_path, status="mechanism_passed_smoke_pending_independent_audit"
    )
    assert validate_training_invocation(args, prereg)[0] == "smoke"
    prereg["prerequisites"]["mechanism"]["status"] = "pending"
    with pytest.raises(RuntimeError):
        validate_training_invocation(args, prereg)
    assert not args.output_dir.exists()

    prereg, args = _stage_prereg(tmp_path / "pilot", status="pilot_pending_independent_audit")
    assert validate_training_invocation(args, prereg)[0] == "pilot"
    prereg["prerequisites"]["smoke"]["status"] = "pending"
    with pytest.raises(RuntimeError):
        validate_training_invocation(args, prereg)
    assert not args.output_dir.exists()


def test_mechanism_path_override_rejected_before_bindings(tmp_path: Path) -> None:
    prereg_path = tmp_path / "prereg.json"
    paths = {
        "preregistration": str(prereg_path),
        "mechanism_selection": str(tmp_path / "selection.json"),
        "cache": str(tmp_path / "cache.h5"),
        "source_h5": str(tmp_path / "source.h5"),
        "r3_best": str(tmp_path / "best.pt"),
        "mechanism_output": str(tmp_path / "report.json"),
    }
    prereg = {
        "schema": "transfer_dg_parent_anchored_k4_preregistration_v1",
        "candidate": "transfer_dg_parent_anchored_k4_r4_20260904",
        "status": "draft_pending_audit",
        "paths": paths,
    }
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    args = argparse.Namespace(
        preregistration=prereg_path,
        selection_manifest=Path(paths["mechanism_selection"]),
        cache=Path(paths["cache"]),
        source_h5=Path(paths["source_h5"]),
        checkpoint=Path(paths["r3_best"]),
        output=tmp_path / "override.json",
    )
    with pytest.raises(RuntimeError, match="path override"):
        validate_mechanism_contract(args, prereg)


def test_failure_payload_contains_safe_available_fields(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    payload = training_failure_payload(
        error=RuntimeError("synthetic"),
        started=0.0,
        output_dir=output,
        input_hashes_before={"code": "hash"},
        preregistration_sha256_before="prereg-hash",
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        argv=["trainer", "--smoke"],
        guard=None,
        peak_allocated_bytes=0,
    )
    assert payload["status"] == "failed"
    assert payload["input_hashes_before"] == {"code": "hash"}
    assert payload["preregistration_sha256_observed_before"] == "prereg-hash"
    assert payload["truth_ledger"] is None
    assert payload["validation_opened"] is False
    assert payload["test_id_opened"] is False


class _LiveChild:
    pid = 4321
    returncode = None

    def poll(self):
        return None


def test_launcher_bounded_mocked_identity_log_and_compute_verification(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    output.mkdir()
    log = tmp_path / "pilot.log"
    prereg_hash = "a" * 64
    (output / "run_identity.json").write_text(
        json.dumps(
            {
                "candidate": "transfer_dg_parent_anchored_k4_r4_20260904",
                "mode": "pilot",
                "preregistration_sha256_observed_before": prereg_hash,
                "output_dir": str(output.resolve()),
            }
        ),
        encoding="utf-8",
    )
    log.write_text('{"event": "k4_identity"}\n', encoding="utf-8")
    ticks = iter([0.0, 0.0])
    evidence = verify_child_launch(
        child=_LiveChild(),
        run_identity=output / "run_identity.json",
        log_path=log,
        candidate="transfer_dg_parent_anchored_k4_r4_20260904",
        preregistration_sha256=prereg_hash,
        output_dir=output,
        query_processes=lambda: [{"pid": "4321", "gpu_uuid": "GPU-x", "used_memory_mib": "10"}],
        monotonic=lambda: next(ticks),
        sleep=lambda seconds: None,
        timeout_seconds=30.0,
    )
    assert evidence["child_alive"]
    assert evidence["compute_pid_verified"]


def test_launcher_schema_status_rejected_before_spawn(tmp_path: Path) -> None:
    preregistration = tmp_path / "prereg.json"
    prereg = {"schema": "wrong", "candidate": "wrong", "status": "wrong"}
    preregistration.write_text(json.dumps(prereg), encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema"):
        validate_launcher_contract(prereg, preregistration)
    prereg = {
        "schema": "transfer_dg_parent_anchored_k4_preregistration_v1",
        "candidate": "transfer_dg_parent_anchored_k4_r4_20260904",
        "status": "draft_pending_audit",
    }
    with pytest.raises(RuntimeError, match="status"):
        validate_launcher_contract(prereg, preregistration)


def test_launcher_exact_child_argv_rejected_before_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    preregistration = tmp_path / "prereg.json"
    mechanism = tmp_path / "mechanism.json"
    smoke = tmp_path / "smoke.json"
    mechanism.write_text("{}\n", encoding="utf-8")
    smoke.write_text("{}\n", encoding="utf-8")
    prereg = {
        "schema": "transfer_dg_parent_anchored_k4_preregistration_v1",
        "candidate": "transfer_dg_parent_anchored_k4_r4_20260904",
        "status": "pilot_pending_independent_audit",
        "paths": {
            "preregistration": str(preregistration),
            "cache": str(tmp_path / "cache.h5"),
            "source_h5": str(tmp_path / "source.h5"),
            "pilot_output_dir": str(tmp_path / "pilot"),
            "pilot_log": str(tmp_path / "pilot.log"),
            "supervisor_identity": str(tmp_path / "supervisor.json"),
        },
        "prerequisites": {
            "mechanism": {"status": "passed", "path": str(mechanism), "sha256": sha256(mechanism)},
            "smoke": {"status": "passed", "path": str(smoke), "sha256": sha256(smoke)},
        },
        "exact_argv": {"pilot_child": "python unexpected.py --pilot"},
    }
    preregistration.write_text(json.dumps(prereg), encoding="utf-8")
    with pytest.raises(RuntimeError, match="argv override"):
        validate_launcher_contract(prereg, preregistration)


def test_baseline_line_digest_uses_original_bytes_without_lf() -> None:
    path = ROOT / "results/transfer_dg_parent_anchored_relenergy_r3_20260904/training_12epoch/metrics.jsonl"
    line3 = path.read_bytes().splitlines()[2]
    assert hashlib.sha256(line3).hexdigest() == "18d301fa4c0cc718d5cdc31e637531c0d62bea028ee21d6327c5a0e4635dd1dc"
    assert hashlib.sha256(line3 + b"\n").hexdigest() != hashlib.sha256(line3).hexdigest()
    event = json.loads(line3)
    assert event["epoch"] == 3 and event["update"] == 204
    assert event["calibration"]["candidate_mean"] == 0.48439965464783963
