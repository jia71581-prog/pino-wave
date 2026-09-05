from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.audit_transfer_dg_r3_exact_gradient_sampling import (
    AUTHORITY_RECORDS_SHA256,
    COMPATIBILITY_RECORDS_SHA256,
    RHO_GRID,
    STARTS,
    apply_normalized_direction,
    assert_complete_state_equal,
    canonical_bytes,
    canonical_sha256,
    capture_complete_state,
    classify,
    cuda_preflight,
    environment_preflight,
    expected_window_length,
    inclusion_mean_errors,
    restore_complete_state,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION = (
    ROOT
    / "results/transfer_dg_r3_exact_gradient_sampling_audit_v1_20260904/selection_manifest.json"
)


def test_canonical_selection_digests_and_serialization_contract() -> None:
    manifest = json.loads(SELECTION.read_text(encoding="utf-8"))
    records = manifest["records"]
    assert canonical_sha256(records) == AUTHORITY_RECORDS_SHA256
    projection = [
        {key: value for key, value in record.items() if key != "cache_position"}
        for record in records
    ]
    assert canonical_sha256(projection) == COMPATIBILITY_RECORDS_SHA256
    assert manifest["digests"]["authority"]["sha256"] == AUTHORITY_RECORDS_SHA256
    assert (
        manifest["digests"]["compatibility_projection"]["sha256"]
        == COMPATIBILITY_RECORDS_SHA256
    )
    encoded = canonical_bytes(records)
    assert not encoded.endswith(b"\n")
    assert b" " not in encoded
    assert encoded.decode("utf-8").startswith("[{")


def test_frozen_starts_cover_last_short_window_and_ht_inclusion() -> None:
    assert STARTS == (2, 34, 66, 98, 130, 162, 194, 226, 258, 290, 322, 354, 386)
    assert [expected_window_length(start) for start in STARTS[:-2]] == [64] * 11
    assert expected_window_length(STARTS[-2]) == 47
    assert expected_window_length(STARTS[-1]) == 15
    errors = inclusion_mean_errors()
    assert errors["frame_max_absolute_error"] <= 1.0e-6
    assert errors["delta_max_absolute_error"] <= 1.0e-6


def _classification_metrics() -> dict:
    return {
        "invalid_failures": [],
        "cosine_full_total_full_main": 0.90,
        "training_window_total_best_main_improvement": 0.02,
        "full_main_best_main_improvement": 0.02,
        "per_record_sampling_cv": [0.5, 0.5, 0.5, 0.5],
        "all_window_to_own_ht_cosine_median": 0.60,
        "cosine_training_window_ht_main_full_main": 0.90,
        "per_record_tbptt_main_cosines": [0.8, 0.8, 0.8, 0.8],
        "window_total_clip_fraction": 0.24,
        "training_window_total_best_composite_relative_change": 1.0e-8,
        "training_window_total_max_per_record_main_regression": 0.01,
    }


def test_classification_boundaries_are_exact() -> None:
    metrics = _classification_metrics()
    assert classify(metrics)["classification"] == "optimization_limited_supported"

    metrics = _classification_metrics()
    metrics["cosine_full_total_full_main"] = 0.80
    assert not classify(metrics)["flags"]["target_or_auxiliary_mismatch"]
    metrics["cosine_full_total_full_main"] = 0.799999
    assert classify(metrics)["classification"] == "target_or_auxiliary_mismatch"

    metrics = _classification_metrics()
    metrics["per_record_sampling_cv"] = [1.0, 1.0, 1.0, 1.0]
    assert not classify(metrics)["flags"]["sampling_limited"]
    metrics["per_record_sampling_cv"] = [1.00001, 1.00001, 1.00001, 0.0]
    assert classify(metrics)["classification"] == "sampling_limited"

    metrics = _classification_metrics()
    metrics["all_window_to_own_ht_cosine_median"] = 0.50
    assert not classify(metrics)["flags"]["sampling_limited"]
    metrics["all_window_to_own_ht_cosine_median"] = 0.499999
    assert classify(metrics)["classification"] == "sampling_limited"

    metrics = _classification_metrics()
    metrics["cosine_training_window_ht_main_full_main"] = 0.80
    metrics["per_record_tbptt_main_cosines"] = [0.70] * 4
    assert not classify(metrics)["flags"]["tbptt_or_rollout_graph_limited"]
    metrics["cosine_training_window_ht_main_full_main"] = 0.799999
    assert classify(metrics)["classification"] == "tbptt_or_rollout_graph_limited"

    metrics = _classification_metrics()
    metrics["window_total_clip_fraction"] = 0.25
    assert classify(metrics)["classification"] == "clip_limited"

    metrics = _classification_metrics()
    metrics["training_window_total_best_main_improvement"] = 0.009999
    assert not classify(metrics)["flags"]["optimization_limited_supported"]

    metrics = _classification_metrics()
    metrics["training_window_total_best_main_improvement"] = 0.0
    metrics["full_main_best_main_improvement"] = 0.000999
    assert classify(metrics)["classification"] == "local_plateau_architecture_or_target_candidate"


def test_classification_priority_and_parallel_flags() -> None:
    metrics = _classification_metrics()
    metrics.update(
        {
            "invalid_failures": ["binding"],
            "cosine_full_total_full_main": 0.1,
            "per_record_sampling_cv": [2.0] * 4,
            "cosine_training_window_ht_main_full_main": 0.1,
            "window_total_clip_fraction": 1.0,
        }
    )
    result = classify(metrics)
    assert result["classification"] == "invalid"
    assert result["flags"]["target_or_auxiliary_mismatch"]
    assert result["flags"]["sampling_limited"]
    assert result["flags"]["tbptt_or_rollout_graph_limited"]
    assert result["flags"]["clip_limited"]


class _TinyStateful(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        self.right = torch.nn.Parameter(torch.tensor([-2.0]))
        self.register_buffer("counter", torch.tensor([7], dtype=torch.int64))


def test_normalized_direction_formula_and_complete_state_rng_rollback() -> None:
    torch.manual_seed(1234)
    module = _TinyStateful()
    module.left.grad = torch.tensor([0.25, -0.50])
    module.right.grad = None
    state = capture_complete_state(module)
    direction = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64)
    rho = RHO_GRID[-1]
    metadata = apply_normalized_direction(module, state, direction, rho)

    theta0 = torch.cat([state.parameters["left"], state.parameters["right"]]).double()
    expected = theta0 - rho * theta0.norm() * direction / direction.norm()
    observed = torch.cat([module.left.detach(), module.right.detach()]).double()
    assert torch.allclose(observed, expected, rtol=1.0e-6, atol=1.0e-7)
    assert metadata["parameter_norm"] == pytest.approx(float(theta0.norm()))
    assert metadata["gradient_norm"] == pytest.approx(float(direction.norm()))

    with torch.no_grad():
        module.counter.add_(1)
    module.left.grad = torch.ones_like(module.left)
    module.right.grad = torch.ones_like(module.right)
    torch.rand(5)
    restore_complete_state(module, state)
    assert_complete_state_equal(module, state)
    assert torch.equal(module.counter, torch.tensor([7]))
    assert torch.equal(module.left.grad, torch.tensor([0.25, -0.50]))
    assert module.right.grad is None


def test_zero_direction_uses_frozen_denominator_floor_and_rolls_back() -> None:
    module = _TinyStateful()
    state = capture_complete_state(module)
    zero = torch.zeros(3, dtype=torch.float64)
    metadata = apply_normalized_direction(module, state, zero, RHO_GRID[-1])
    assert metadata["gradient_norm"] == 0.0
    assert torch.equal(module.left.detach(), state.parameters["left"])
    assert torch.equal(module.right.detach(), state.parameters["right"])
    restore_complete_state(module, state)
    assert_complete_state_equal(module, state)


def _patch_successful_cuda_preflight(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append("set_device"))
    monkeypatch.setattr(torch.cuda, "init", lambda: calls.append("init"))
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append("reset_peak_memory_stats"),
    )
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled: calls.append("deterministic_algorithms"),
    )
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append("torch_seed"))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append("cuda_seed"))
    monkeypatch.setattr(
        "scripts.audit_transfer_dg_r3_exact_gradient_sampling.np.random.seed",
        lambda seed: calls.append("numpy_seed"),
    )


def _synthetic_preregistration(tmp_path: Path) -> Path:
    path = tmp_path / "preregistration.json"
    path.write_text('{"synthetic":true}\n', encoding="utf-8")
    return path


def test_cuda_preflight_initialization_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    _patch_successful_cuda_preflight(monkeypatch, calls)
    preregistration = _synthetic_preregistration(tmp_path)
    result = cuda_preflight(
        device=torch.device("cuda:0"),
        seed=372,
        output=tmp_path / "must_not_exist.json",
        candidate="synthetic",
        mode="smoke",
        input_hashes_before={"synthetic": "bound"},
        preregistration=preregistration,
        preregistration_sha256_before=canonical_sha256({"synthetic": "unused"}),
        environment={"synthetic": "whitelist"},
        after_success=lambda: calls.append("after_success") or "continued",
    )
    assert result == "continued"
    assert calls[:3] == ["set_device", "init", "reset_peak_memory_stats"]
    assert calls[3:] == [
        "deterministic_algorithms",
        "torch_seed",
        "cuda_seed",
        "numpy_seed",
        "after_success",
    ]
    assert not (tmp_path / "must_not_exist.json").exists()


@pytest.mark.parametrize("failure_call", ["init", "reset_peak_memory_stats"])
def test_cuda_preflight_failure_is_atomic_sealed_and_stops_loaders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_call: str
) -> None:
    calls: list[str] = []
    loaders = {"cache": 0, "model": 0}

    def set_device(device) -> None:
        calls.append("set_device")

    def init() -> None:
        calls.append("init")
        if failure_call == "init":
            raise RuntimeError("synthetic init failure")

    def reset(device) -> None:
        calls.append("reset_peak_memory_stats")
        if failure_call == "reset_peak_memory_stats":
            raise RuntimeError("synthetic reset failure")

    def loaders_after_success() -> None:
        loaders["cache"] += 1
        loaders["model"] += 1

    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch.cuda, "init", init)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    preregistration = _synthetic_preregistration(tmp_path)
    output = tmp_path / f"{failure_call}.json"
    with pytest.raises(RuntimeError, match="synthetic"):
        cuda_preflight(
            device=torch.device("cuda:0"),
            seed=372,
            output=output,
            candidate="synthetic",
            mode="smoke",
            input_hashes_before={"synthetic": "bound"},
            preregistration=preregistration,
            preregistration_sha256_before=canonical_sha256({"synthetic": "unused"}),
            environment={"synthetic": "whitelist"},
            after_success=loaders_after_success,
        )
    expected_calls = ["set_device", "init"]
    if failure_call == "reset_peak_memory_stats":
        expected_calls.append("reset_peak_memory_stats")
    assert calls == expected_calls
    assert loaders == {"cache": 0, "model": 0}
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["classification"]["classification"] == "invalid"
    assert payload["classification"]["invalid_failures"] == [
        "cuda_initialization_failed"
    ]
    assert payload["failure_stage"] == "cuda_initialization_failed"
    assert payload["diagnostic_entered"] is False
    assert payload["promotion_authorized"] is False
    assert payload["validation_opened"] is False
    assert payload["test_id_opened"] is False
    assert "preregistration_sha256_observed_before" in payload
    assert "preregistration_sha256_observed_after" in payload


@pytest.mark.parametrize(
    ("cublas", "visible", "expected_failure"),
    [
        (None, "0", "CUBLAS_WORKSPACE_CONFIG"),
        (":16:8", "0", "CUBLAS_WORKSPACE_CONFIG"),
        (":4096:8", "1", "CUDA_VISIBLE_DEVICES"),
    ],
)
def test_environment_preflight_invalid_before_cuda_or_loaders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cublas: str | None,
    visible: str,
    expected_failure: str,
) -> None:
    if cublas is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", cublas)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-reported")
    calls = {"cuda_preflight": 0, "cache_loader": 0, "model_loader": 0}

    def forbidden_continuation() -> None:
        calls["cuda_preflight"] += 1
        calls["cache_loader"] += 1
        calls["model_loader"] += 1

    preregistration = _synthetic_preregistration(tmp_path)
    output = tmp_path / "environment_failure.json"
    with pytest.raises(RuntimeError, match="environment preflight failed"):
        environment_preflight(
            preregistration=preregistration,
            output=output,
            candidate="synthetic",
            mode="smoke",
            after_success=forbidden_continuation,
        )
    assert calls == {"cuda_preflight": 0, "cache_loader": 0, "model_loader": 0}
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["classification"]["classification"] == "invalid"
    assert payload["classification"]["invalid_failures"] == [
        "environment_preflight_failed"
    ]
    assert payload["failure_stage"] == "environment_preflight_failed"
    assert expected_failure in payload["environment_contract_failures"]
    assert payload["diagnostic_entered"] is False
    assert payload["promotion_authorized"] is False
    assert payload["validation_opened"] is False
    assert payload["test_id_opened"] is False
    assert payload["preregistration_hash_unchanged"] is True
    assert (
        payload["preregistration_sha256_observed_before"]
        == payload["preregistration_sha256_observed_after"]
    )
    assert set(payload["observed_environment"]) == {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "python",
        "torch",
        "torch_cuda",
        "cudnn",
        "numpy",
        "h5py",
    }
    assert "UNRELATED_SECRET" not in payload["observed_environment"]
    assert "must-not-be-reported" not in output.read_text(encoding="utf-8")


def test_environment_preflight_exact_contract_enters_cuda_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls: list[str] = []
    preregistration = _synthetic_preregistration(tmp_path)
    observed, preregistration_hash = environment_preflight(
        preregistration=preregistration,
        output=tmp_path / "must_not_exist.json",
        candidate="synthetic",
        mode="smoke",
        after_success=lambda: calls.append("cuda_preflight"),
    )
    assert calls == ["cuda_preflight"]
    assert observed["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert observed["CUDA_VISIBLE_DEVICES"] == "0"
    assert len(preregistration_hash) == 64
    assert not (tmp_path / "must_not_exist.json").exists()
