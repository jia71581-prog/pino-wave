# Elastic VTI Component-Balanced Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Elastic VTI 训练增加向后兼容的分量均衡 relative L2，并验证它能降低 Marmousi `u_x` 误差且不损害 `u_z`。

**Architecture:** 新 helper 按最后一维组件分别归一化 relative L2；未配置组件权重时调用原 `combined_loss`，保持所有现有任务行为。普通监督、adaptive balancing 和空间重要性采样路径统一调用 Elastic 监督 wrapper；空间路径扩展为同时支持四维声学和五维双分量张量。

**Tech Stack:** Python 3.10、PyTorch、pytest、CUDA、YAML、nohup。

---

## File Map

- Modify `src/fno_acoustic/train_elastic_vti.py`: 分量损失、监督 wrapper、五维空间抽样支持和诊断。
- Create `tests/test_elastic_component_balanced_loss.py`: 数学、默认兼容、非法配置及空间路径测试。
- Create `configs/pino_elastic_vti_marmousi_component_balanced_short.yaml`: 5-epoch 候选配置。
- Create `tests/test_marmousi_component_balanced_config.py`: 候选配置契约。
- Generate `artifacts/elastic_vti_pino_marmousi_component_balanced_short/`: 候选训练和评估产物。
- Generate only after acceptance `configs/pino_elastic_vti_marmousi_component_balanced_finetune.yaml` and formal artifacts.

### Task 1: Define Component-Loss Behavior With Failing Tests

**Files:**
- Create: `tests/test_elastic_component_balanced_loss.py`

- [ ] **Step 1: Add mathematical behavior tests**

```python
import pytest
import torch

from fno_acoustic import train_elastic_vti as training


def test_component_balanced_relative_l2_equalizes_component_amplitude() -> None:
    target = torch.ones(1, 2, 2, 3, 2)
    target[..., 1] *= 10.0
    pred = target.clone()
    pred[..., 0] += 0.5
    pred[..., 1] += 1.0

    loss, components = training._component_balanced_relative_l2(
        pred,
        target,
        component_weights=[1.0, 1.0],
        eps=1.0e-8,
    )

    assert loss.item() == pytest.approx(0.3, abs=1.0e-6)
    assert components.tolist() == pytest.approx([0.5, 0.1], abs=1.0e-6)


@pytest.mark.parametrize("weights", [[1.0], [1.0, 0.0], [1.0, float("nan")]])
def test_component_balanced_relative_l2_rejects_invalid_weights(weights) -> None:
    values = torch.ones(1, 2, 2, 3, 2)
    with pytest.raises(ValueError):
        training._component_balanced_relative_l2(values, values, weights, eps=1.0e-8)
```

- [ ] **Step 2: Add backward-compatibility test**

```python
def test_elastic_supervised_loss_without_component_weights_matches_legacy() -> None:
    torch.manual_seed(7)
    pred = torch.randn(2, 3, 4, 5, 2)
    target = torch.randn_like(pred)
    config = {"relative_l2_weight": 1.0, "mse_weight": 0.05, "eps": 1.0e-8}

    actual, _ = training._elastic_supervised_loss(pred, target, config)
    expected, _ = training.combined_loss(pred, target, **config)

    assert torch.allclose(actual, expected)
```

- [ ] **Step 3: Run tests and verify missing helpers fail**

Run: `PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_elastic_component_balanced_loss.py`

Expected: failures because `_component_balanced_relative_l2` and `_elastic_supervised_loss` do not exist.

### Task 2: Implement Component-Balanced Supervised Loss

**Files:**
- Modify: `src/fno_acoustic/train_elastic_vti.py`
- Test: `tests/test_elastic_component_balanced_loss.py`

- [ ] **Step 1: Implement component validation and loss**

Add near `_per_sample_relative_l2`:

```python
def _component_balanced_relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    component_weights: list[float] | tuple[float, ...],
    eps: float,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {tuple(pred.shape)} != {tuple(target.shape)}")
    if pred.ndim < 3:
        raise ValueError("component-balanced loss expects batch and component dimensions")
    weights = torch.as_tensor(component_weights, device=pred.device, dtype=pred.dtype)
    if weights.ndim != 1 or weights.numel() != pred.shape[-1]:
        raise ValueError("component weights must match the final component dimension")
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("component weights must be finite and positive")
    diff2 = (pred - target).pow(2)
    target2 = target.pow(2)
    if sample_weights is not None:
        diff2 = diff2 * sample_weights
        target2 = target2 * sample_weights
    reduce_dims = tuple(range(1, pred.ndim - 1))
    numerator = torch.sqrt(diff2.sum(dim=reduce_dims).clamp_min(0.0))
    denominator = torch.sqrt(target2.sum(dim=reduce_dims).clamp_min(0.0)).clamp_min(float(eps))
    per_component = numerator / denominator
    per_sample = (per_component * weights.view(1, -1)).sum(dim=-1) / weights.sum()
    return per_sample.mean(), per_component.mean(dim=0)
```

- [ ] **Step 2: Implement the backward-compatible wrapper**

```python
def _elastic_supervised_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    config = _supervised_loss_config(loss_config)
    component_weights = config.pop("component_relative_l2_weights", None)
    if component_weights is None:
        return combined_loss(pred, target, **config)
    relative_weight = float(config.pop("relative_l2_weight", 1.0))
    base, parts = combined_loss(pred, target, relative_l2_weight=0.0, **config)
    balanced, components = _component_balanced_relative_l2(
        pred, target, component_weights, eps=float(config.get("eps", 1.0e-8))
    )
    parts["relative_l2_global"] = parts.pop("relative_l2")
    parts["relative_l2"] = float(balanced.detach().cpu())
    for index, value in enumerate(components.detach().cpu().tolist()):
        parts[f"relative_l2_component_{index}"] = float(value)
    return base + relative_weight * balanced, parts
```

Add `component_relative_l2_weights` to `SUPERVISED_LOSS_KEYS`.

- [ ] **Step 3: Replace ordinary and adaptive-balancing calls**

Replace all Elastic training calls of:

```python
combined_loss(pred, y, **_supervised_loss_config(config["loss"]))
```

with:

```python
_elastic_supervised_loss(pred, y, config["loss"])
```

- [ ] **Step 4: Run component tests**

Run: `PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_elastic_component_balanced_loss.py tests/test_elastic_vti_operator_learning.py`

Expected: all tests pass.

### Task 3: Support Five-Dimensional Spatial Sampling

**Files:**
- Modify: `src/fno_acoustic/train_elastic_vti.py`
- Modify: `tests/test_elastic_component_balanced_loss.py`

- [ ] **Step 1: Add failing full-sampling equivalence test**

```python
def test_spatial_importance_loss_matches_full_component_loss_when_all_pixels_selected() -> None:
    torch.manual_seed(9)
    pred = torch.randn(1, 2, 3, 4, 2)
    target = torch.randn_like(pred)
    indices = torch.arange(6)
    probabilities = torch.full((6,), 1.0 / 6.0)
    config = {
        "relative_l2_weight": 1.0,
        "mse_weight": 0.0,
        "component_relative_l2_weights": [1.0, 1.0],
        "eps": 1.0e-8,
    }
    sampled, _ = training._spatial_importance_loss(
        pred, target, indices, probabilities, 6, config, reweight=True
    )
    full, _ = training._elastic_supervised_loss(pred, target, config)
    assert torch.allclose(sampled, full, atol=1.0e-6)


def test_spatial_residual_map_supports_elastic_components() -> None:
    values = torch.zeros(2, 3, 4, 5, 2)
    result = training._spatial_residual_map(values + 1.0, values)
    assert result.shape == (3, 4)
    assert torch.allclose(result, torch.ones_like(result))
```

- [ ] **Step 2: Verify tests fail on current four-dimensional implementation**

Expected: tensor unpacking and dimension validation failures.

- [ ] **Step 3: Generalize flattening and residual map**

For `_spatial_residual_map`, accept rank 4 or 5 and average rank 5 over `(batch,time,component)`.

For `_spatial_importance_loss`, preserve rank 4 behavior and reshape rank 5 as `[B,H*W,T,C]`. Reshape inverse-probability weights to `[1,P,1]` or `[1,P,1,1]`. When component weights are configured, call `_component_balanced_relative_l2` with those sample weights; otherwise preserve the existing global weighted relative L2.

- [ ] **Step 4: Run focused loss tests**

Run: `PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_elastic_component_balanced_loss.py tests/test_train_step.py -k 'spatial or supervised_loss_config'`

Expected: all selected tests pass.

### Task 4: Add And Validate Short Candidate Config

**Files:**
- Create: `configs/pino_elastic_vti_marmousi_component_balanced_short.yaml`
- Create: `tests/test_marmousi_component_balanced_config.py`

- [ ] **Step 1: Write failing config contract test**

Require baseline `best.pt`, frozen split/stats, `[1.0,1.0]` component weights, 5 epochs, 64 train batches, 8 validation batches, `lr=2e-5`, `weight_decay=1e-5`, `grad_clip=0.5`, and independent output directories.

- [ ] **Step 2: Create config from the `lr2e5` candidate**

Apply:

```yaml
train:
  epochs: 5
  max_train_batches: 64
  max_val_batches: 8
  checkpoint_dir: artifacts/elastic_vti_pino_marmousi_component_balanced_short/checkpoints
  log_dir: artifacts/elastic_vti_pino_marmousi_component_balanced_short/logs
loss:
  component_relative_l2_weights: [1.0, 1.0]
evaluation:
  output_dir: artifacts/elastic_vti_pino_marmousi_component_balanced_short/evaluation
```

- [ ] **Step 3: Run all new and Elastic baseline tests**

Run: `PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_elastic_component_balanced_loss.py tests/test_marmousi_component_balanced_config.py tests/test_elastic_vti_operator_learning.py`

Expected: all pass.

- [ ] **Step 4: Commit implementation**

Commit only the modified Elastic training module, new tests and candidate config.

### Task 5: Train, Evaluate And Gate Candidate

**Files:**
- Generate: `artifacts/elastic_vti_pino_marmousi_component_balanced_short/`.

- [ ] **Step 1: Verify baseline hash and GPU capacity**

Require baseline SHA256 `7f05716e87075718df9291ca76aa5aaacb85e071c6e35d819d5ef4762c54efd8` and at least 10 GiB free VRAM. Do not stop unrelated tasks without explicit need.

- [ ] **Step 2: Run 5-epoch candidate in foreground**

Use `scripts/train_elastic_vti_pino.py`, the candidate config, CUDA, independent `train.log`, and `train_metrics.json`.

Expected: exit 0, finite ratio 1.0, five metric rows, checkpoint reload succeeds.

- [ ] **Step 3: Generate fixed four-sample component evaluation**

Run `scripts/evaluate_current_training_visuals.py` for validation indices `3,23,40,48` and the candidate `best.pt`.

- [ ] **Step 4: Apply all gates**

Require:

```text
u_x <= 0.13161056
u_z <= 0.08560451
global validation relative L2 <= 0.1049973499
prediction_finite_ratio == 1.0
```

Write the decision and measured values to `acceptance.json` before any formal submission.

### Task 6: Submit Formal Finetune Only If Accepted

**Files:**
- Generate conditionally: formal config, PID and nohup log.

- [ ] **Step 1: Stop on rejection**

If any gate fails, do not generate or submit formal training; preserve baseline and report the failed gates.

- [ ] **Step 2: Create formal config on acceptance**

Reset initialization to the frozen original baseline, set 40 epochs, full train batches, early stopping patience 10 and min delta `1e-4`, and independent artifact directories.

- [ ] **Step 3: Submit with nohup and perform startup-only verification**

Write PID/log, wait 10 seconds, verify process alive and no OOM/NaN. Stop monitoring after startup verification.

### Task 7: Final Verification

- [ ] **Step 1: Run focused tests and task-file whitespace checks**

Run all new tests plus `tests/test_elastic_vti_operator_learning.py`; run `git diff --check` only on task files because the worktree has a pre-existing paper whitespace failure.

- [ ] **Step 2: Report evidence**

Report changed files, commit, candidate training/validation metrics, component metrics, acceptance decision, artifacts, and whether formal nohup was submitted. Do not claim full-suite success because the existing suite has unrelated failures and a multiprocessing deadlock.

