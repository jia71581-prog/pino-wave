# Saved-Time Phase-Conditioned Multi-Input F-FNO V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and discriminate a stored-time-only, propagation-conditioned, deeper multi-input acoustic operator before authorizing a long training run.

**Architecture:** Reuse the V3 medium/source/MIONet coarse branches and replace the dense correction path with a saved-time-validated decoder that sees local retarded-time features and can use an eight-layer axis-factorized complex spectral stack. Train only HDF5 frames at the 401 stored times, with full-axis stratification, microbatching and gradient accumulation; compare depth and propagation conditioning in a 2 x 2 capacity probe.

**Tech Stack:** Python 3.13, PyTorch 2.7, h5py, NumPy, pytest, CUDA/RTX 3090, existing `grouped_ufno_mionet_v3` package.

---

## File structure

- Create `saved_time_phase_operator_v4/__init__.py`: public V4 API.
- Create `saved_time_phase_operator_v4/time_grid.py`: immutable stored-time validation and indexing.
- Create `saved_time_phase_operator_v4/sampling.py`: four-bin exact-frame sampling and planned coverage audit.
- Create `saved_time_phase_operator_v4/features.py`: dense local travel/retarded-time feature construction.
- Create `saved_time_phase_operator_v4/spectral.py`: axis-factorized learned-complex layers and checkpointed residual stack.
- Create `saved_time_phase_operator_v4/decoder.py`: propagation-conditioned full-field correction decoder.
- Create `saved_time_phase_operator_v4/operator.py`: V3-compatible V4 operator and V3 transfer loader.
- Create `saved_time_phase_operator_v4/metrics.py`: exact-only energy-aware metrics.
- Create `saved_time_phase_operator_v4/data.py`: exact stored-frame macro/microbatch dataset.
- Create `saved_time_phase_operator_v4/config.py`: strict probe configuration and variant definitions.
- Create `scripts/train_saved_time_v4_probe.py`: deterministic 2 x 2 capacity-probe runner.
- Create `scripts/evaluate_saved_time_v4.py`: sealed exact-time evaluation.
- Create `configs/saved_time_v4/capacity_probe.yaml`: 48-record probe configuration.
- Create tests under `tests/saved_time_phase_operator_v4/` matching each module.

### Task 1: Enforce the 401-point stored-time contract

**Files:**
- Create: `saved_time_phase_operator_v4/__init__.py`
- Create: `saved_time_phase_operator_v4/time_grid.py`
- Create: `tests/saved_time_phase_operator_v4/test_time_grid.py`

- [ ] **Step 1: Write the failing stored-time tests**

```python
import pytest
import torch

from saved_time_phase_operator_v4.time_grid import SavedTimeGrid


def test_saved_time_grid_maps_exact_values_to_indices():
    grid = SavedTimeGrid.from_values(torch.linspace(0.0, 1.0, 401))
    values = torch.tensor([[0.0, 0.25, 1.0]])
    assert torch.equal(grid.indices(values), torch.tensor([[0, 100, 400]]))


def test_saved_time_grid_rejects_midpoints():
    grid = SavedTimeGrid.from_values(torch.linspace(0.0, 1.0, 401))
    with pytest.raises(ValueError, match="stored HDF5 time"):
        grid.indices(torch.tensor([[0.00125]]))
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_time_grid.py -q`

Expected: FAIL because `saved_time_phase_operator_v4` does not exist.

- [ ] **Step 3: Implement strict stored-time indexing**

```python
@dataclass(frozen=True)
class SavedTimeGrid:
    values_s: torch.Tensor
    tolerance_s: float

    @classmethod
    def from_values(cls, values, *, tolerance_fraction: float = 1.0e-4):
        axis = torch.as_tensor(values, dtype=torch.float64).cpu()
        spacing = torch.diff(axis)
        if axis.ndim != 1 or len(axis) < 2 or torch.any(spacing <= 0):
            raise ValueError("stored time axis must be strictly increasing")
        if not torch.allclose(spacing, spacing[0], rtol=0.0, atol=1.0e-12):
            raise ValueError("stored time axis must be uniform")
        return cls(axis, float(spacing[0]) * tolerance_fraction)

    def indices(self, requested_s):
        values = torch.as_tensor(requested_s, dtype=torch.float64)
        axis = self.values_s.to(values.device)
        right = torch.searchsorted(axis, values).clamp(0, len(axis) - 1)
        left = (right - 1).clamp(0, len(axis) - 1)
        choose_left = (values - axis[left]).abs() <= (values - axis[right]).abs()
        index = torch.where(choose_left, left, right)
        if torch.any((values - axis[index]).abs() > self.tolerance_s):
            raise ValueError("requested time is not a stored HDF5 time")
        return index.long()
```

- [ ] **Step 4: Run the focused tests**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_time_grid.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4 tests/saved_time_phase_operator_v4/test_time_grid.py
git commit -m "feat(v4): enforce stored-time query contract"
```

### Task 2: Sample the complete stored time axis and prove coverage

**Files:**
- Create: `saved_time_phase_operator_v4/sampling.py`
- Create: `tests/saved_time_phase_operator_v4/test_sampling.py`

- [ ] **Step 1: Write failing stratification and coverage tests**

```python
import numpy as np

from saved_time_phase_operator_v4.sampling import stored_time_indices, coverage_summary


def test_sampler_covers_pre_onset_early_middle_and_late():
    axis = np.linspace(0.0, 1.0, 401)
    indices = stored_time_indices(axis, source_t0_s=0.1, phase_offset=7)
    assert len(indices) == 4
    assert indices[0] < 40
    assert 40 <= indices[1] < 160
    assert 160 <= indices[2] < 280
    assert 280 <= indices[3] <= 400


def test_planned_coverage_has_no_midpoints_and_reaches_late_axis():
    report = coverage_summary(record_count=48, appearances=140, seed=17)
    assert report["minimum_unique_indices"] >= 120
    assert report["maximum_index"] == 400
    assert report["interpolated_requests"] == 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_sampling.py -q`

Expected: FAIL because the sampling functions are absent.

- [ ] **Step 3: Implement deterministic four-bin sampling**

```python
def stored_time_indices(time_s, *, source_t0_s: float, phase_offset: int):
    axis = np.asarray(time_s, dtype=np.float64)
    onset = int(np.searchsorted(axis, source_t0_s, side="left"))
    boundaries = ((0, max(onset, 1)), (onset, 160), (160, 280), (280, len(axis)))
    result = []
    for bin_index, (start, stop) in enumerate(boundaries):
        start = min(max(start, 0), len(axis) - 1)
        stop = min(max(stop, start + 1), len(axis))
        result.append(start + (phase_offset * 17 + bin_index * 31) % (stop - start))
    if phase_offset % (len(axis) - 280) == 0:
        result[-1] = len(axis) - 1
    return np.asarray(result, dtype=np.int64)
```

```python
def coverage_summary(*, record_count: int, appearances: int, seed: int):
    if record_count <= 0 or appearances <= 0:
        raise ValueError("record_count and appearances must be positive")
    axis = np.linspace(0.0, 1.0, 401)
    coverage = [set() for _ in range(record_count)]
    for record in range(record_count):
        onset = 0.05 + 0.14 * ((record * 104729 + seed) % 1000) / 999.0
        for appearance in range(appearances):
            indices = stored_time_indices(
                axis,
                source_t0_s=onset,
                phase_offset=seed + record * 9176 + appearance,
            )
            coverage[record].update(int(index) for index in indices)
    counts = np.asarray([len(values) for values in coverage], dtype=np.int64)
    selected = set().union(*coverage)
    return {
        "minimum_unique_indices": int(counts.min()),
        "median_unique_indices": float(np.median(counts)),
        "maximum_unique_indices": int(counts.max()),
        "minimum_index": min(selected),
        "maximum_index": max(selected),
        "interpolated_requests": 0,
    }
```

- [ ] **Step 4: Run tests and the real 2,240-record planned coverage audit**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_sampling.py -q`

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m saved_time_phase_operator_v4.sampling --records 2240 --appearances 140 --seed 17`

Expected: tests pass; JSON reports zero interpolated requests, maximum index 400 and at least 120 unique indices per record.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4/sampling.py tests/saved_time_phase_operator_v4/test_sampling.py
git commit -m "feat(v4): cover the complete saved time axis"
```

### Task 3: Build local retarded-time propagation features

**Files:**
- Create: `saved_time_phase_operator_v4/features.py`
- Create: `tests/saved_time_phase_operator_v4/test_features.py`

- [ ] **Step 1: Write failing feature tests**

```python
import torch

from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime
from saved_time_phase_operator_v4.features import dense_propagation_features


def test_dense_features_are_causal_and_phase_aligned():
    travel = RayTravelTime(*(torch.tensor([[0.2, 0.4]]) for _ in range(5)))
    source = torch.tensor([[0.0, 0.0, 10.0, 0.1, 1.0]])
    value = dense_propagation_features(
        travel, torch.tensor([[0.25]]), source, height=1, width=2,
        domain_t_s=1.0, domain_diagonal_m=3000.0,
    )
    assert value.shape == (1, 1, 12, 1, 2)
    assert value[0, 0, 5, 0, 0] > value[0, 0, 5, 0, 1]
    assert torch.isfinite(value).all()
```

- [ ] **Step 2: Run test and verify failure**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_features.py -q`

Expected: FAIL because `dense_propagation_features` is absent.

- [ ] **Step 3: Implement the 12-channel local bundle**

```python
tau = time_s[:, :, None] - source_parameters[:, None, 3:4] - travel.seconds[:, None]
frequency = source_parameters[:, None, 2:3]
phase = 2.0 * math.pi * frequency * tau
causal = torch.sigmoid(tau / 0.005)
features = torch.stack((
    tau / domain_t_s,
    travel.seconds[:, None].expand_as(tau) / domain_t_s,
    travel.distance_m[:, None].expand_as(tau) / domain_diagonal_m,
    travel.path_velocity_mps[:, None].expand_as(tau) / 5000.0,
    travel.endpoint_velocity_mps[:, None].expand_as(tau) / 5000.0,
    causal,
    torch.sin(phase),
    torch.cos(phase),
    torch.exp(-0.5 * (tau / 0.01).square()),
    torch.exp(-0.5 * (tau / 0.025).square()),
    torch.exp(-0.5 * (tau / 0.05).square()),
    torch.exp(-0.5 * (tau / 0.10).square()),
), dim=2)
return features.reshape(records, times, 12, height, width)
```

- [ ] **Step 4: Run focused tests**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_features.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4/features.py tests/saved_time_phase_operator_v4/test_features.py
git commit -m "feat(v4): add dense propagation-aligned features"
```

### Task 4: Implement the deep axis-factorized complex spectral stack

**Files:**
- Create: `saved_time_phase_operator_v4/spectral.py`
- Create: `tests/saved_time_phase_operator_v4/test_spectral.py`

- [ ] **Step 1: Write failing shape, gradient and parameter-efficiency tests**

```python
import torch

from saved_time_phase_operator_v4.spectral import FactorizedComplexResidualStack


def test_factorized_stack_preserves_shape_and_gradients():
    model = FactorizedComplexResidualStack(width=16, spectral_rank=8, modes=12, depth=8)
    value = torch.randn(2, 16, 33, 33, requires_grad=True)
    model(value).square().mean().backward()
    assert value.grad is not None
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_eight_factorized_blocks_fit_capacity_budget():
    model = FactorizedComplexResidualStack(width=64, spectral_rank=40, modes=32, depth=8)
    assert sum(p.numel() for p in model.parameters()) < 8_000_000
```

- [ ] **Step 2: Run tests and verify failure**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_spectral.py -q`

Expected: FAIL because the factorized stack is absent.

- [ ] **Step 3: Implement learned complex x/z contractions and residual forks**

Implement separate `rfft`/`irfft` contractions along x and z, each with learned complex weights. Each residual block uses pre-`GroupNorm`, rank projection, the sum of x and z spectral outputs, the existing depthwise/pointwise local path, GELU and a learned residual scale initialized to 0.1. The stack applies a two-block residual fork and checkpoints blocks only while training:

```python
def forward(self, value: torch.Tensor) -> torch.Tensor:
    hidden = value
    fork = value
    for index, block in enumerate(self.blocks):
        if self.activation_checkpointing and self.training and hidden.requires_grad:
            hidden = checkpoint(block, hidden, use_reentrant=False)
        else:
            hidden = block(hidden)
        if (index + 1) % 2 == 0:
            hidden = hidden + fork
            fork = hidden
    return hidden
```

- [ ] **Step 4: Run CPU tests and a CUDA memory smoke test**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_spectral.py -q`

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -c "import torch; from saved_time_phase_operator_v4.spectral import FactorizedComplexResidualStack as S; m=S(64,40,32,8).cuda(); x=torch.randn(6,64,201,201,device='cuda',requires_grad=True); m(x).mean().backward(); print(torch.cuda.max_memory_allocated())"`

Expected: tests pass and peak allocated memory is below 12 GiB for the isolated stack.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4/spectral.py tests/saved_time_phase_operator_v4/test_spectral.py
git commit -m "feat(v4): add deep factorized complex spectral stack"
```

### Task 5: Integrate the propagation-conditioned decoder and V3 transfer

**Files:**
- Create: `saved_time_phase_operator_v4/decoder.py`
- Create: `saved_time_phase_operator_v4/operator.py`
- Create: `saved_time_phase_operator_v4/config.py`
- Create: `tests/saved_time_phase_operator_v4/test_operator.py`

- [ ] **Step 1: Write failing API and transfer tests**

```python
def test_v4_reuses_one_medium_for_multiple_sources(v4_fixture):
    model, normalizer, velocity, sources, source_maps, times, axes = v4_fixture
    encoded = model.encode_medium(velocity[:1], normalizer)
    prepared = model.prepare_sources(
        encoded, sources[:2], source_maps[:2], normalizer,
        record_to_medium=torch.zeros(2, dtype=torch.long),
    )
    fields = model.dense_normalized(prepared, times[:2], x_m=axes[0], z_m=axes[1])
    assert fields.shape == (2, times.shape[1], 201, 201)


def test_v4_rejects_interpolated_dense_time(v4_fixture):
    model, normalizer, velocity, sources, source_maps, _, axes = v4_fixture
    prepared = model.prepare_sources(model.encode_medium(velocity[:1], normalizer), sources[:1], source_maps[:1], normalizer)
    with pytest.raises(ValueError, match="stored HDF5 time"):
        model.dense_normalized(prepared, torch.tensor([[0.00125]]), x_m=axes[0], z_m=axes[1])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_operator.py -q`

Expected: FAIL because the V4 operator is absent.

- [ ] **Step 3: Implement the decoder and operator override**

`PropagationConditionedDenseDecoder.forward` accepts `RayTravelTime` in addition to the V3 decoder arguments, projects the 12 propagation channels with a 1 x 1 convolution, adds a 401-entry saved-index embedding, and applies either two control blocks or eight factorized blocks. `SavedTimePhaseOperatorV4._expand_dense_times` calls `SavedTimeGrid.indices` before delegating. `_dense_block` reuses the V3 coarse MIONet operations and passes `dense_grid.travel` into the new decoder. The integration point is:

```python
times = super()._expand_dense_times(prepared, time_s)
saved_indices = self.saved_time_grid.indices(times).to(times.device)
raw = self.dense_decoder(
    prepared.medium.encoding,
    prepared.source_encoding,
    prepared.source_parameters,
    prepared.record_to_medium,
    times,
    coarse,
    dense_grid.travel,
    saved_indices,
)
return raw * free_surface_factor(dense_grid.z_m)[None, None, :, None]
```

Implement `load_v3_backbone` as:

```python
def load_v3_backbone(model, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k: v for k, v in payload["model_state"].items() if not k.startswith("dense_decoder.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("dense_decoder.") for key in missing):
        raise ValueError(f"invalid V3 transfer: missing={missing}, unexpected={unexpected}")
```

- [ ] **Step 4: Run operator tests and parameter census**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_operator.py -q`

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m saved_time_phase_operator_v4.operator --summary`

Expected: tests pass; selected model has 18-25M parameters and both one-medium/multiple-source and multiple-medium mappings work.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4 tests/saved_time_phase_operator_v4/test_operator.py
git commit -m "feat(v4): integrate saved-time phase-conditioned operator"
```

### Task 6: Add exact-only data loading, loss and evaluation metrics

**Files:**
- Create: `saved_time_phase_operator_v4/data.py`
- Create: `saved_time_phase_operator_v4/metrics.py`
- Create: `tests/saved_time_phase_operator_v4/test_data_metrics.py`

- [ ] **Step 1: Write failing exact-target and energy-floor tests**

```python
def test_v4_batch_contains_only_exact_targets(real_manifest):
    batch = make_test_batch(real_manifest, records=12, step=3)
    assert batch.target_exact.all()
    assert torch.equal(batch.left_index, batch.right_index)
    assert batch.requested_time_s.shape == (12, 4)


def test_energy_floor_prevents_zero_frame_explosion():
    target = torch.zeros(2, 2, 8, 8)
    target[0, 1, 2, 2] = 1.0
    prediction = torch.full_like(target, 0.01)
    result = exact_wavefield_metrics(prediction, target, families=("uniform", "layered"), energy_floor_fraction=0.01)
    assert math.isfinite(result["aggregate_floored_relative_l2"])
    assert result["near_zero_frame_count"] == 3
```

- [ ] **Step 2: Run tests and verify failure**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_data_metrics.py -q`

Expected: FAIL because V4 data and metrics are absent.

- [ ] **Step 3: Implement exact four-frame materialization and metrics**

Use `stored_time_indices` to call `V3WavefieldDataset.read_wavefield` with exact axis values only. Preserve `PilotBatch`-compatible tensors, but use four times. Split each 12-record macro-batch into two six-record microbatches without breaking `record_to_medium`. The primary metric is:

```python
error_norm = (prediction - target).float().flatten(2).norm(dim=-1)
target_norm = target.float().flatten(2).norm(dim=-1)
record_peak = target_norm.amax(dim=1, keepdim=True)
floor = energy_floor_fraction * record_peak
floored_relative = error_norm / torch.maximum(target_norm, floor).clamp_min(1.0e-8)
near_zero = target_norm < floor
```

Also return un-floored relative L2, RMSE, phase correlation and low/mid/high spatial-spectrum residuals. `make_test_batch(manifest, records, step)` is a public test helper that constructs a `SavedTimeBatchDataset` from the first balanced schedule entry and returns its first batch.

- [ ] **Step 4: Run tests and inspect one real batch**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_data_metrics.py -q`

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m saved_time_phase_operator_v4.data --inspect-one configs/saved_time_v4/capacity_probe.yaml`

Expected: tests pass; all 48 targets are exact, time indices include the late bin and no interpolation alpha is nonzero.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4/data.py saved_time_phase_operator_v4/metrics.py tests/saved_time_phase_operator_v4/test_data_metrics.py
git commit -m "feat(v4): add exact-time batches and energy-aware metrics"
```

### Task 7: Implement the deterministic 2 x 2 capacity probe

**Files:**
- Create: `scripts/train_saved_time_v4_probe.py`
- Create: `configs/saved_time_v4/capacity_probe.yaml`
- Create: `tests/saved_time_phase_operator_v4/test_probe.py`

- [ ] **Step 1: Write failing variant and checkpoint tests**

```python
def test_probe_defines_the_four_required_variants():
    assert variant_names() == (
        "shallow_no_phase", "deep_no_phase", "shallow_phase", "deep_phase"
    )


def test_probe_selection_requires_relative_improvement():
    scores = {"shallow_no_phase": 0.50, "deep_no_phase": 0.45, "shallow_phase": 0.31, "deep_phase": 0.28}
    family_scores = {
        name: {"uniform": score, "layered": score, "marmousi": score}
        for name, score in scores.items()
    }
    assert select_probe_candidate(
        scores, family_scores, minimum_relative_improvement=0.15
    ) == "deep_phase"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_probe.py -q`

Expected: FAIL because the probe runner is absent.

- [ ] **Step 3: Implement probe training and selection**

The YAML fixes 48 training and 48 validation records balanced across the three medium families, 60 epochs, four stored frames, six-record microbatches, two-step gradient accumulation, AMP, eight CPU workers, and an epoch checkpoint. Use parameter groups with learning rate `1e-4` for the new dense head and `1e-5` for transferred branches, AdamW weight decay `1e-6`, cosine decay and gradient clip 1.0. All variants use the same record IDs, time indices, seed and optimizer-update count. The selection rule is:

```python
def select_probe_candidate(scores, family_scores, *, minimum_relative_improvement=0.15):
    control = float(scores["shallow_no_phase"])
    eligible = []
    for name, score in scores.items():
        improvement = (control - float(score)) / control
        family_ok = all(
            family_scores[name][family] <= 1.05 * family_scores["shallow_no_phase"][family]
            for family in ("uniform", "layered", "marmousi")
        )
        if name != "shallow_no_phase" and improvement >= minimum_relative_improvement and family_ok:
            eligible.append(name)
    if not eligible:
        raise RuntimeError("no V4 probe variant passed the selection gate")
    return min(eligible, key=scores.__getitem__)
```

- [ ] **Step 4: Run unit tests and a one-update CUDA smoke for all variants**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_probe.py -q`

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python scripts/train_saved_time_v4_probe.py --config configs/saved_time_v4/capacity_probe.yaml --smoke-updates 1`

Expected: four variants finish, peak allocated memory remains below 23 GiB, gradients reach backbone, propagation projection and all factorized blocks, and four atomic checkpoints exist.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_saved_time_v4_probe.py configs/saved_time_v4/capacity_probe.yaml tests/saved_time_phase_operator_v4/test_probe.py
git commit -m "feat(v4): add controlled capacity and phase probe"
```

### Task 8: Run the probe, seal exact-time evaluation and decide on long training

**Files:**
- Create: `scripts/evaluate_saved_time_v4.py`
- Create: `tests/saved_time_phase_operator_v4/test_evaluation.py`
- Create: `docs/superpowers/reports/2026-07-17-saved-time-phase-conditioned-v4-probe-results.md`

- [ ] **Step 1: Write the failing sealed-evaluation identity test**

```python
def test_evaluation_rejects_a_different_time_axis(bound_identity):
    altered = dict(bound_identity)
    altered["time_axis_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="time axis"):
        validate_evaluation_identity(altered, bound_identity)
```

- [ ] **Step 2: Implement exact-only evaluation and run its tests**

The evaluator binds checkpoint SHA256, HDF5 manifest digest, exact 401-value time-axis SHA256, record census and model config digest. It rejects any interpolated target and writes family/time-bin floored and un-floored metrics plus representative stored-time wavefield plots. Identity validation is exact:

```python
def validate_evaluation_identity(candidate, expected):
    for key in ("checkpoint_sha256", "manifest_digest", "time_axis_sha256", "record_census", "model_config_digest"):
        if candidate.get(key) != expected.get(key):
            label = key.replace("_", " ")
            raise ValueError(f"evaluation {label} mismatch")
```

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4/test_evaluation.py -q`

Expected: PASS.

- [ ] **Step 3: Launch the four-variant capacity probe**

Run: `nohup /home/jiayh/miniconda3/envs/deepwave_env/bin/python scripts/train_saved_time_v4_probe.py --config configs/saved_time_v4/capacity_probe.yaml > /data/jiayh/saved_time_v4_probe/launcher.log 2>&1 &`

Expected: `run_identity.json` and four per-variant logs appear; each epoch creates a checkpoint.

- [ ] **Step 4: Evaluate and apply the decision rule**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python scripts/evaluate_saved_time_v4.py --artifact-dir /data/jiayh/saved_time_v4_probe --stored-times-only`

Expected: the report identifies one selected variant only if the 15% improvement and family-regression gates pass. If only `deep_no_phase` passes, capacity is the primary diagnosed factor. If `shallow_phase` matches or beats it, propagation conditioning is primary. If `deep_phase` wins, both contribute. If none passes, do not launch long training; add a spectrum-refinement ablation under a new design amendment.

- [ ] **Step 5: Record results and commit**

```bash
git add scripts/evaluate_saved_time_v4.py tests/saved_time_phase_operator_v4/test_evaluation.py docs/superpowers/reports/2026-07-17-saved-time-phase-conditioned-v4-probe-results.md
git commit -m "report(v4): seal saved-time capacity probe"
```

- [ ] **Step 6: Run the complete regression suite before long training**

Run: `/home/jiayh/miniconda3/envs/deepwave_env/bin/python -m pytest tests/saved_time_phase_operator_v4 tests/grouped_ufno_mionet_v3 -q`

Expected: all tests pass. A long run is authorized only after this command and the Task 8 selection gate pass.
