# DCLP-NO Two-Snapshot LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the frozen epoch-40 DCLP-NO independently for each instance using only the two complete fields nearest \(t_0+2/f_0\) and \(t_0+4/f_0\), then evaluate every later frame and 37 near-surface receivers in a sealed process.

**Architecture:** Inject zero-initialized rank-4 LoRA wrappers into the local-field and dense-corrector time, phase, source, and map projections. A new non-adjacent snapshot guard exposes only the two approved fields; adaptation writes a hashed adapter and access audit before a separate evaluator is allowed to open future truth.

**Tech Stack:** Python 3.13, PyTorch, h5py, NumPy, PyYAML, pytest, SHA-256.

---

## File Map

- Create `saved_time_phase_operator_v4/instance_adaptation/lora.py`: wrappers, injection, adapter state.
- Create `saved_time_phase_operator_v4/instance_adaptation/cycle_data_guard.py`: strict two-cycle data view.
- Create `saved_time_phase_operator_v4/instance_adaptation/lora_loss.py`: two-field-only loss.
- Create `saved_time_phase_operator_v4/instance_adaptation/lora_trainer.py`: deterministic optimization and observed-only gate.
- Create `saved_time_phase_operator_v4/instance_adaptation/lora_sealing.py`: atomic adapter sealing and verification.
- Create `configs/tgrs_dclp_no/lora_rank4.yaml`: search space and fixed primary layer targets.
- Create `scripts/adapt_tgrs_two_snapshot_lora.py`: adaptation-only process.
- Create `scripts/evaluate_tgrs_two_snapshot_lora.py`: sealed future evaluator.
- Create `scripts/select_tgrs_lora_hyperparameters.py`: validation-only selection and frozen resolved config.
- Create five focused test files under `tests/saved_time_phase_operator_v4/`.

### Task 1: Implement zero-initialized LoRA wrappers and exact layer injection

**Files:**
- Create: `tests/saved_time_phase_operator_v4/test_tgrs_lora.py`
- Create: `saved_time_phase_operator_v4/instance_adaptation/lora.py`
- Modify: `saved_time_phase_operator_v4/instance_adaptation/__init__.py`

- [ ] **Step 1: Write failing identity and freezing tests**

```python
from __future__ import annotations

import torch
from torch import nn

from saved_time_phase_operator_v4.instance_adaptation.lora import (
    LoRAConv1x1,
    LoRALinear,
    adapter_state_dict,
    inject_lora,
)


class ToyOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_field = nn.Module()
        self.local_field.source_projection = nn.Linear(4, 4)
        self.local_field.phase_projection = nn.Conv2d(12, 4, 1)
        self.dense_decoder = nn.Module()
        self.dense_decoder.source_projection = nn.Linear(4, 4)


def test_linear_and_conv_lora_start_at_exact_parent_function():
    torch.manual_seed(3)
    linear = nn.Linear(4, 5)
    conv = nn.Conv2d(3, 4, 1)
    wrapped_linear = LoRALinear(linear, rank=2, alpha=2.0)
    wrapped_conv = LoRAConv1x1(conv, rank=2, alpha=2.0)
    x = torch.randn(2, 4)
    image = torch.randn(2, 3, 5, 5)
    assert torch.equal(wrapped_linear(x), linear(x))
    assert torch.equal(wrapped_conv(image), conv(image))


def test_injection_freezes_parent_and_exposes_only_adapter_parameters():
    model = ToyOperator()
    targets = (
        "local_field.source_projection",
        "local_field.phase_projection",
        "dense_decoder.source_projection",
    )
    manifest = inject_lora(model, targets=targets, rank=4, alpha=4.0)
    assert tuple(manifest["targets"]) == targets
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if ".base." in name)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(".lora_down." in name or ".lora_up." in name for name in trainable)
    assert set(adapter_state_dict(model)) == trainable
```

- [ ] **Step 2: Verify failure**

Run:

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_lora.py
```

Expected: import failure for `instance_adaptation.lora`.

- [ ] **Step 3: Implement the wrappers**

Create `saved_time_phase_operator_v4/instance_adaptation/lora.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float):
        super().__init__()
        if int(rank) <= 0 or float(alpha) <= 0.0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_down = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.normal_(self.lora_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.scale * self.lora_up(self.lora_down(value))


class LoRAConv1x1(nn.Module):
    def __init__(self, base: nn.Conv2d, *, rank: int, alpha: float):
        super().__init__()
        if base.kernel_size != (1, 1) or base.groups != 1:
            raise ValueError("LoRAConv1x1 requires an ungrouped 1x1 convolution")
        if int(rank) <= 0 or float(alpha) <= 0.0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_down = nn.Conv2d(base.in_channels, self.rank, 1, bias=False)
        self.lora_up = nn.Conv2d(self.rank, base.out_channels, 1, bias=False)
        nn.init.normal_(self.lora_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.scale * self.lora_up(self.lora_down(value))


def _replace(model: nn.Module, target: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = target.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, replacement)


def inject_lora(
    model: nn.Module,
    *,
    targets: Sequence[str],
    rank: int,
    alpha: float,
) -> dict[str, object]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    resolved = []
    for target in tuple(str(value) for value in targets):
        module = model.get_submodule(target)
        if isinstance(module, nn.Linear):
            wrapped = LoRALinear(module, rank=rank, alpha=alpha)
        elif isinstance(module, nn.Conv2d):
            wrapped = LoRAConv1x1(module, rank=rank, alpha=alpha)
        else:
            raise TypeError(f"unsupported LoRA target {target}: {type(module).__name__}")
        _replace(model, target, wrapped)
        resolved.append(target)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable == 0:
        raise ValueError("LoRA injection produced no trainable parameters")
    return {
        "schema": "dclp_no_lora_injection_v1",
        "targets": tuple(resolved),
        "rank": int(rank),
        "alpha": float(alpha),
        "trainable_parameters": int(trainable),
    }


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (".lora_down." in name or ".lora_up." in name)
    }


def load_adapter_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    expected = {
        name for name, parameter in named.items()
        if parameter.requires_grad
        and (".lora_down." in name or ".lora_up." in name)
    }
    if set(state) != expected:
        raise ValueError("adapter state keys do not match injected LoRA parameters")
    with torch.no_grad():
        for name in sorted(expected):
            named[name].copy_(state[name].to(device=named[name].device, dtype=named[name].dtype))
```

- [ ] **Step 4: Export and test**

Add to `instance_adaptation/__init__.py`:

```python
from .lora import (
    LoRAConv1x1,
    LoRALinear,
    adapter_state_dict,
    inject_lora,
    load_adapter_state_dict,
)
```

Run:

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_lora.py
```

Expected: `2 passed`.

### Task 2: Add a non-adjacent two-cycle data guard

**Files:**
- Create: `tests/saved_time_phase_operator_v4/test_tgrs_cycle_data_guard.py`
- Create: `saved_time_phase_operator_v4/instance_adaptation/cycle_data_guard.py`

- [ ] **Step 1: Write the guard tests**

```python
from __future__ import annotations

import pytest

from saved_time_phase_operator_v4.instance_adaptation.cycle_data_guard import (
    TwoCycleAccessAudit,
)


def test_two_cycle_audit_accepts_only_the_registered_nonadjacent_pair():
    audit = TwoCycleAccessAudit((80, 120))
    assert audit.read((80, 120)) == (80, 120)
    assert audit.payload() == {
        "allowed_indices": (80, 120),
        "requested_indices": (80, 120),
        "future_truth_used": False,
    }
    with pytest.raises(PermissionError, match="future truth"):
        audit.read((121,))


def test_two_cycle_audit_rejects_adjacent_or_unordered_protocols():
    with pytest.raises(ValueError, match="nonadjacent"):
        TwoCycleAccessAudit((3, 4))
    with pytest.raises(ValueError, match="ordered"):
        TwoCycleAccessAudit((5, 2))
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_cycle_data_guard.py
```

Expected: import failure for `cycle_data_guard`.

- [ ] **Step 3: Implement the audit and guarded record**

Create `cycle_data_guard.py` by reusing the read-only V3 record path from
`data_guard.py`, with these public definitions:

```python
class TwoCycleAccessAudit:
    def __init__(self, allowed_indices):
        first, second = tuple(int(value) for value in allowed_indices)
        if first < 0 or second <= first:
            raise ValueError("two-cycle indices must be ordered and nonnegative")
        if second == first + 1:
            raise ValueError("two-cycle protocol must be nonadjacent")
        self.allowed_indices = (first, second)
        self.requested_indices = []

    def read(self, indices):
        requested = tuple(int(value) for value in indices)
        for value in requested:
            if value not in self.requested_indices:
                self.requested_indices.append(value)
        if any(value not in self.allowed_indices for value in requested):
            raise PermissionError("future truth access is forbidden during adaptation")
        return requested

    def payload(self):
        return {
            "allowed_indices": self.allowed_indices,
            "requested_indices": tuple(self.requested_indices),
            "future_truth_used": any(
                value not in self.allowed_indices for value in self.requested_indices
            ),
        }
```

Add frozen dataclass `GuardedCycleRecord` with the same target-free fields as
`GuardedOnsetRecord`, plus:

```python
observed_indices: tuple[int, int]
observed_times_s: tuple[float, float]
observed_wavefield: torch.Tensor
audit: TwoCycleAccessAudit
```

Add `GuardedCycleDataset`. In `__getitem__`, compute:

```python
observed = two_cycle_observation_indices(
    metadata_record.time_s,
    t0_s=float(metadata_record.source_parameters[3]),
    f0_hz=float(metadata_record.source_parameters[2]),
)
audit = TwoCycleAccessAudit(observed)
audit.read(observed)
frames = np.asarray(
    h5["wavefield"][metadata.source_index, list(observed), :, :],
    dtype=np.float32,
)
```

Use the existing `EikonalTravelCache`, `_digest_tensors`, `close`, and context
manager pattern from `data_guard.py`. Do not alter `GuardedOnsetDataset`.

- [ ] **Step 4: Add a fake-HDF5 dataset test**

Extend `test_tgrs_cycle_data_guard.py` with the monkeypatched
`V3WavefieldDataset` pattern in
`tests/saved_time_phase_operator_v4/test_clfc_data_guard.py`. Use:

```python
time_s = torch.arange(401, dtype=torch.float32) * 0.0025
source_parameters = torch.tensor([1000.0, 1000.0, 20.0, 0.10, 1.0])
```

Assert indices `(80, 120)`, field shape `(2, 4, 4)`, and no future access.

- [ ] **Step 5: Run the guard tests**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_cycle_data_guard.py
```

Expected: all tests pass.

### Task 3: Implement a two-field-only loss

**Files:**
- Create: `tests/saved_time_phase_operator_v4/test_tgrs_lora_loss.py`
- Create: `saved_time_phase_operator_v4/instance_adaptation/lora_loss.py`

- [ ] **Step 1: Write failing loss tests**

```python
from __future__ import annotations

import torch

from saved_time_phase_operator_v4.instance_adaptation.lora_loss import (
    TwoSnapshotLossWeights,
    two_snapshot_loss,
)


def test_identical_two_snapshot_fields_have_zero_data_terms():
    field = torch.randn(1, 2, 16, 16)
    terms = two_snapshot_loss(
        field,
        field,
        adapter_parameters=(torch.zeros(4, requires_grad=True),),
        weights=TwoSnapshotLossWeights(),
    )
    for name in ("field", "gradient", "spectrum_phase"):
        assert float(terms[name]) < 1.0e-7


def test_loss_uses_exactly_two_frames():
    prediction = torch.zeros(1, 3, 8, 8)
    target = torch.zeros_like(prediction)
    try:
        two_snapshot_loss(prediction, target, adapter_parameters=())
    except ValueError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("three frames were accepted")
```

- [ ] **Step 2: Verify failure**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_lora_loss.py
```

Expected: import failure for `lora_loss`.

- [ ] **Step 3: Implement the loss**

Create `lora_loss.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TwoSnapshotLossWeights:
    field: float = 1.0
    gradient: float = 0.15
    spectrum_phase: float = 0.10
    correction: float = 1.0e-6


def _relative_norm(error: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return error.flatten(1).norm(dim=1).div(
        target.detach().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    ).mean()


def two_snapshot_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    adapter_parameters,
    weights: TwoSnapshotLossWeights = TwoSnapshotLossWeights(),
) -> dict[str, torch.Tensor]:
    predicted = torch.as_tensor(prediction).float()
    reference = torch.as_tensor(target, device=predicted.device).float()
    if predicted.shape != reference.shape or predicted.ndim != 4 or predicted.shape[1] != 2:
        raise ValueError("LoRA loss requires exactly two matching complete fields")
    field = _relative_norm(predicted - reference, reference)
    predicted_grad = torch.cat(
        (
            (predicted[..., 1:, :] - predicted[..., :-1, :]).flatten(1),
            (predicted[..., :, 1:] - predicted[..., :, :-1]).flatten(1),
        ),
        dim=1,
    )
    reference_grad = torch.cat(
        (
            (reference[..., 1:, :] - reference[..., :-1, :]).flatten(1),
            (reference[..., :, 1:] - reference[..., :, :-1]).flatten(1),
        ),
        dim=1,
    )
    gradient = (
        (predicted_grad - reference_grad).norm(dim=1)
        / reference_grad.detach().norm(dim=1).clamp_min(1.0e-8)
    ).mean()
    predicted_fft = torch.fft.rfft2(predicted, norm="ortho")
    reference_fft = torch.fft.rfft2(reference, norm="ortho")
    spectrum_phase = (
        (predicted_fft - reference_fft).abs().flatten(1).norm(dim=1)
        / reference_fft.abs().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    ).mean()
    parameters = tuple(adapter_parameters)
    correction = (
        predicted.sum() * 0.0
        if not parameters
        else torch.cat([value.reshape(-1) for value in parameters]).square().mean()
    )
    total = (
        float(weights.field) * field
        + float(weights.gradient) * gradient
        + float(weights.spectrum_phase) * spectrum_phase
        + float(weights.correction) * correction
    )
    return {
        "field": field,
        "gradient": gradient,
        "spectrum_phase": spectrum_phase,
        "correction": correction,
        "total": total,
    }
```

- [ ] **Step 4: Run the loss tests**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_lora_loss.py
```

Expected: `2 passed`.

### Task 4: Add observed-only optimization and rollback

**Files:**
- Create: `tests/saved_time_phase_operator_v4/test_tgrs_lora_trainer.py`
- Create: `saved_time_phase_operator_v4/instance_adaptation/lora_trainer.py`

- [ ] **Step 1: Write a toy adaptation test**

```python
from __future__ import annotations

import torch
from torch import nn

from saved_time_phase_operator_v4.instance_adaptation.lora_trainer import (
    LoRAAdaptationConfig,
    adapt_two_snapshots,
)


class ToyLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_up = nn.Parameter(torch.zeros(()))


def test_trainer_reads_only_registered_fields_and_improves_observed_loss():
    model = ToyLoRA()
    target = torch.ones(1, 2, 4, 4)
    requested = []

    def predict_observed():
        requested.extend((8, 16))
        return model.lora_up.expand_as(target)

    result = adapt_two_snapshots(
        model,
        target,
        predict_observed=predict_observed,
        accessed_indices=lambda: tuple(dict.fromkeys(requested)),
        observed_indices=(8, 16),
        config=LoRAAdaptationConfig(steps=20, learning_rate=0.1),
    )
    assert result.accepted
    assert result.accessed_true_indices == (8, 16)
    assert result.candidate_observed_loss < result.parent_observed_loss
```

- [ ] **Step 2: Verify failure**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_lora_trainer.py
```

Expected: import failure for `lora_trainer`.

- [ ] **Step 3: Implement deterministic adaptation**

Create `lora_trainer.py` with dataclasses:

```python
@dataclass(frozen=True)
class LoRAAdaptationConfig:
    steps: int = 100
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    gradient_clip: float = 1.0
    maximum_observed_correction_ratio: float = 0.35
    seed: int = 372
    weights: TwoSnapshotLossWeights = TwoSnapshotLossWeights()


@dataclass(frozen=True)
class LoRAAdaptationResult:
    accepted: bool
    rollback_reason: str | None
    steps: int
    parent_observed_loss: float
    candidate_observed_loss: float
    observed_correction_ratio: float
    accessed_true_indices: tuple[int, ...]
    future_truth_used: bool
    elapsed_s: float
    trainable_parameter_count: int
```

Implement `adapt_two_snapshots` as:

```python
def adapt_two_snapshots(
    model,
    target_observed,
    *,
    predict_observed,
    accessed_indices,
    observed_indices,
    config=LoRAAdaptationConfig(),
):
    started = time.perf_counter()
    torch.manual_seed(int(config.seed))
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("LoRA adaptation has no trainable parameters")
    target = torch.as_tensor(target_observed, device=parameters[0].device).float()
    with torch.no_grad():
        parent = predict_observed().detach()
        parent_terms = two_snapshot_loss(
            parent, target, adapter_parameters=parameters, weights=config.weights
        )
    initial = adapter_state_dict(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    failure = None
    for _ in range(int(config.steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_observed()
        terms = two_snapshot_loss(
            prediction, target, adapter_parameters=parameters, weights=config.weights
        )
        if not bool(torch.isfinite(terms["total"])):
            failure = "nonfinite_loss"
            break
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(config.gradient_clip))
        optimizer.step()
    with torch.no_grad():
        candidate = predict_observed().detach()
        candidate_terms = two_snapshot_loss(
            candidate, target, adapter_parameters=parameters, weights=config.weights
        )
        correction_ratio = float(
            (candidate - parent).norm() / parent.norm().clamp_min(1.0e-8)
        )
    requested = tuple(accessed_indices())
    if requested != tuple(observed_indices):
        failure = "access_audit_failed"
    elif not bool(torch.isfinite(candidate).all()):
        failure = "nonfinite_prediction"
    elif float(candidate_terms["total"]) >= float(parent_terms["total"]):
        failure = "observed_loss_not_improved"
    elif correction_ratio > float(config.maximum_observed_correction_ratio):
        failure = "observed_correction_too_large"
    accepted = failure is None
    if not accepted:
        load_adapter_state_dict(model, initial)
    return LoRAAdaptationResult(
        accepted=accepted,
        rollback_reason=failure,
        steps=int(config.steps),
        parent_observed_loss=float(parent_terms["total"]),
        candidate_observed_loss=float(candidate_terms["total"]),
        observed_correction_ratio=correction_ratio,
        accessed_true_indices=requested,
        future_truth_used=requested != tuple(observed_indices),
        elapsed_s=time.perf_counter() - started,
        trainable_parameter_count=sum(value.numel() for value in parameters),
    )
```

Import `time`, `dataclass`, `torch`,
`adapter_state_dict`, `load_adapter_state_dict`,
`TwoSnapshotLossWeights`, and `two_snapshot_loss`.

- [ ] **Step 4: Run trainer tests**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_lora_trainer.py
```

Expected: all tests pass.

### Task 5: Seal adapters before future-truth access

**Files:**
- Create: `tests/saved_time_phase_operator_v4/test_tgrs_lora_sealing.py`
- Create: `saved_time_phase_operator_v4/instance_adaptation/lora_sealing.py`

- [ ] **Step 1: Write tamper and future-mutation tests**

The test creates `adapter.pt`, hashes it, seals:

```python
payload = {
    "schema": "dclp_no_two_snapshot_lora_v1",
    "sample_id": "test:uniform:00000",
    "observed_indices": [80, 120],
    "access_audit": {
        "allowed_indices": [80, 120],
        "requested_indices": [80, 120],
        "future_truth_used": False,
    },
    "adapter_sha256": sha256_file(adapter_path),
}
```

Assert verification succeeds, then modify `adapter.pt` and assert
`verify_lora_state` raises `ValueError("adapter digest mismatch")`. A second test
creates two HDF5 arrays whose future frames differ but whose two observed frames
match; adapt both and assert identical adapter SHA-256 and sealed state digest.

- [ ] **Step 2: Implement `seal_lora_state` and `verify_lora_state`**

Use `canonical_json_bytes`, `write_json_atomic`, and `sha256_file`. Validation
must require:

```python
observed = tuple(int(value) for value in payload["observed_indices"])
allowed = tuple(int(value) for value in payload["access_audit"]["allowed_indices"])
requested = tuple(int(value) for value in payload["access_audit"]["requested_indices"])
if len(observed) != 2 or observed[1] <= observed[0] + 1:
    raise ValueError("sealed LoRA requires the nonadjacent two-cycle protocol")
if allowed != observed or requested != observed:
    raise ValueError("sealed LoRA access audit mismatch")
if bool(payload["access_audit"]["future_truth_used"]):
    raise ValueError("sealed LoRA used future truth")
if sha256_file(adapter_path) != str(payload["adapter_sha256"]):
    raise ValueError("adapter digest mismatch")
```

The sealed document contains `payload` and
`state_digest=sha256(canonical_json_bytes(payload)).hexdigest()`.

- [ ] **Step 3: Run sealing tests**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_lora_sealing.py
```

Expected: all tests pass.

### Task 6: Add the primary LoRA configuration and adaptation-only CLI

**Files:**
- Create: `configs/tgrs_dclp_no/lora_rank4.yaml`
- Create: `scripts/adapt_tgrs_two_snapshot_lora.py`
- Create: `tests/saved_time_phase_operator_v4/test_tgrs_lora_cli.py`

- [ ] **Step 1: Add the configuration**

```yaml
schema: dclp_no_two_snapshot_lora_search_v1
protocol_config: configs/tgrs_dclp_no/protocol.yaml
parent_operator_config: /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/artifacts/parent_epoch40_smoke3/local_parent_loader.yaml
rank: 4
alpha: 4.0
targets:
  - local_field.source_projection
  - local_field.map_projection
  - local_field.time_mlp.0
  - local_field.time_mlp.2
  - local_field.phase_projection
  - dense_decoder.source_projection
  - dense_decoder.map_projection
  - dense_decoder.time_mlp.0
  - dense_decoder.time_mlp.2
  - dense_decoder.phase_projection
search:
  learning_rates: [0.0005, 0.001, 0.002]
  steps: [50, 100, 200]
fixed:
  weight_decay: 0.000001
  gradient_clip: 1.0
  maximum_observed_correction_ratio: 0.35
  loss:
    field: 1.0
    gradient: 0.15
    spectrum_phase: 0.10
    correction: 0.000001
seed: 372
```

- [ ] **Step 2: Add a dry-run CLI test**

Call `main()` with `--dry-run`, one registered sample, and a temporary output.
Assert printed JSON has:

```python
{
    "future_truth_opened": False,
    "allowed_true_snapshot_count": 2,
    "rank": 4,
    "target_count": 10,
}
```

- [ ] **Step 3: Implement the adaptation-only CLI**

The CLI arguments are:

```python
parser.add_argument("--config", required=True)
parser.add_argument("--resolved-hyperparameters")
parser.add_argument("--split", choices=("validation", "test_id"), required=True)
parser.add_argument("--sample-id", action="append")
parser.add_argument("--manifest")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--dry-run", action="store_true")
```

For each selected record:

1. load the parent with `_load_parent` from
   `scripts/run_v5_instance_adaptation.py`;
2. construct `GuardedCycleDataset`;
3. inject rank-4 LoRA after loading the parent;
4. prepare the medium/source/dense-grid state once;
5. define `predict_observed()` that queries only
   `record.time_s[list(record.observed_indices)]`;
6. call `adapt_two_snapshots`;
7. atomically save only `adapter_state_dict(model)` and non-target metadata;
8. seal `adapter.json`;
9. close the guarded dataset.

The script must not import or call `_read_future_truth`. Add:

```python
if "_read_future_truth" in globals():
    raise RuntimeError("adaptation-only CLI cannot bind a future-truth reader")
```

The sealed payload records sample/group/family/source index, observed
indices/times, access audit, parent checkpoint SHA-256, dataset manifest digest,
resolved hyperparameter digest, adapter SHA-256, parameter count, acceptance,
rollback reason, and elapsed time.

- [ ] **Step 4: Run CLI tests and CPU dry run**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_lora_cli.py
/home/jiayh/miniconda3/bin/python scripts/adapt_tgrs_two_snapshot_lora.py \
  --config configs/tgrs_dclp_no/lora_rank4.yaml \
  --split validation \
  --sample-id validation_uniform_00003 \
  --output-dir artifacts/tgrs_dclp_no/lora/dry_run \
  --device cpu \
  --dry-run
```

Expected: tests pass and dry-run JSON reports no future access.

### Task 7: Add the independent sealed evaluator

**Files:**
- Create: `scripts/evaluate_tgrs_two_snapshot_lora.py`
- Create: `tests/saved_time_phase_operator_v4/test_tgrs_lora_evaluator.py`

- [ ] **Step 1: Write evaluator-order tests**

Monkeypatch `verify_lora_state` and the HDF5 future reader to append events.
Assert the order is exactly:

```python
["verify_adapter", "open_future_truth"]
```

Also assert an invalid digest prevents the future reader from being called.

- [ ] **Step 2: Implement the evaluator**

The evaluator accepts:

```python
parser.add_argument("--config", required=True)
parser.add_argument("--resolved-hyperparameters", required=True)
parser.add_argument("--adapter-dir", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--device", default="cuda")
```

For every sealed adapter:

```python
sealed = verify_lora_state(adapter_json, adapter_pt)
truth = read_full_truth_after_verification(source_h5, sealed.payload["source_index"])
```

Then reload the bound parent, inject the same targets/rank, load the adapter,
and query only `future_indices(401, observed_indices)`. Save parent and adapted
future fields as float32 arrays, raw per-record metrics, timing, memory, and
receiver traces under the sample directory. The evaluator never rewrites the
adapter files.

- [ ] **Step 3: Run evaluator tests**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_tgrs_lora_evaluator.py
```

Expected: all tests pass.

### Task 8: Select global hyperparameters on validation only

**Files:**
- Create: `scripts/select_tgrs_lora_hyperparameters.py`
- Generate: `artifacts/tgrs_dclp_no/lora/resolved_lora.yaml`

- [ ] **Step 1: Run a three-family validation sweep**

For each of nine `(learning_rate, steps)` combinations, adapt the frozen
three-family figure manifest in separate directories. Run future evaluation
only on validation records.

```bash
/home/jiayh/miniconda3/bin/python scripts/select_tgrs_lora_hyperparameters.py \
  --config configs/tgrs_dclp_no/lora_rank4.yaml \
  --validation-manifest artifacts/tgrs_dclp_no/manifests/figure_cases.json \
  --output-dir artifacts/tgrs_dclp_no/lora/hyperparameter_sweep \
  --resolved-output artifacts/tgrs_dclp_no/lora/resolved_lora.yaml
```

Selection key, in order:

1. lowest mean validation future full-field relative L2;
2. lowest receiver cross-correlation lag;
3. lowest adapter parameter norm.

The resolved file freezes rank, alpha, targets, learning rate, steps, loss
weights, correction limit, seed, input config SHA-256, and selection rows.

- [ ] **Step 2: Verify the resolved configuration does not contain test metrics**

Run:

```bash
rg -n "test|test_metric|test_error" artifacts/tgrs_dclp_no/lora/resolved_lora.yaml
```

Expected: no matches.

### Task 9: Run smoke, validation, and test-ID stages

- [ ] **Step 1: Run all focused tests**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q \
  tests/saved_time_phase_operator_v4/test_tgrs_lora.py \
  tests/saved_time_phase_operator_v4/test_tgrs_cycle_data_guard.py \
  tests/saved_time_phase_operator_v4/test_tgrs_lora_loss.py \
  tests/saved_time_phase_operator_v4/test_tgrs_lora_trainer.py \
  tests/saved_time_phase_operator_v4/test_tgrs_lora_sealing.py \
  tests/saved_time_phase_operator_v4/test_tgrs_lora_cli.py \
  tests/saved_time_phase_operator_v4/test_tgrs_lora_evaluator.py
```

Expected: all tests pass.

- [ ] **Step 2: Verify parent equality before optimization**

Run a one-record GPU smoke with `steps=0` and compare the injected model against
the parent. Require:

```text
maximum_absolute_difference = 0
relative_l2_difference = 0
parent_parameter_hash_before = parent_parameter_hash_after
```

- [ ] **Step 3: Run three-family validation pilot**

Use the resolved configuration. Require finite outputs, exactly two accessed
indices, zero future access, valid seals, no free-surface violation, and a
reported rollback decision for every sample.

- [ ] **Step 4: Run all 480 test adaptations**

Adaptation command:

```bash
/home/jiayh/miniconda3/bin/python scripts/adapt_tgrs_two_snapshot_lora.py \
  --config configs/tgrs_dclp_no/lora_rank4.yaml \
  --resolved-hyperparameters artifacts/tgrs_dclp_no/lora/resolved_lora.yaml \
  --split test_id \
  --manifest artifacts/tgrs_dclp_no/manifests/test_records.json \
  --output-dir artifacts/tgrs_dclp_no/lora/test480/adapters \
  --device cuda
```

Evaluation command:

```bash
/home/jiayh/miniconda3/bin/python scripts/evaluate_tgrs_two_snapshot_lora.py \
  --config configs/tgrs_dclp_no/lora_rank4.yaml \
  --resolved-hyperparameters artifacts/tgrs_dclp_no/lora/resolved_lora.yaml \
  --adapter-dir artifacts/tgrs_dclp_no/lora/test480/adapters \
  --output-dir artifacts/tgrs_dclp_no/lora/test480/evaluation \
  --device cuda
```

Expected: 480 verified adapter states and 480 future-evaluation rows. No record
may be replaced based on its result.

- [ ] **Step 5: Refresh the source checkpoint**

```bash
/home/jiayh/miniconda3/bin/python scripts/audit_tgrs_dclp_no.py \
  --config configs/tgrs_dclp_no/protocol.yaml \
  --source-root . \
  --output artifacts/tgrs_dclp_no/audit/source_manifest.json
```

Expected: exit code 0.
