# Grouped Dual-Head Wave Operator V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, gate, and launch a numerically normalized dual-head single-source acoustic operator that accurately predicts full wavefields and arbitrary receiver queries, with FWI explicitly out of scope.

**Architecture:** A new `grouped_ufno_mionet_v2` package uses train-only physical normalization, a shared U-FNO medium/source encoder, a continuous MIONet-attention query head, and a time-conditioned multiscale dense decoder. A structured V2 cache supplies coherent spatial frames and fixed receiver traces; training jointly supervises both heads, audits gradients and loss dominance, and cannot launch production until an eight-record overfit gate and bounded validation pilot beat a zero predictor.

**Tech Stack:** Python 3.13, PyTorch, h5py, NumPy, PyYAML, matplotlib, pytest, CUDA AMP/BF16 where supported.

---

## File map

- `grouped_ufno_mionet_v2/config.py`: strict V2 model/data/loss/train configuration.
- `grouped_ufno_mionet_v2/normalization.py`: train-only velocity/source/pressure scaling and checkpoint metadata.
- `grouped_ufno_mionet_v2/data/cache.py`: V2 cache schema validation and causal HDF5 reader.
- `grouped_ufno_mionet_v2/data/batch.py`: grouped one-source macro-batches with source maps and structured targets.
- `grouped_ufno_mionet_v2/model/medium.py`: normalized multiscale U-FNO medium encoder.
- `grouped_ufno_mionet_v2/model/source.py`: parameter, local-medium, and source-map encoder.
- `grouped_ufno_mionet_v2/model/query.py`: continuous normalized-pressure MIONet-attention head.
- `grouped_ufno_mionet_v2/model/dense.py`: time-conditioned multiscale dense decoder.
- `grouped_ufno_mionet_v2/model/operator.py`: shared encoding and public dual-head API.
- `grouped_ufno_mionet_v2/losses.py`: normalized data, coherent spectral, gradient, and consistency losses.
- `grouped_ufno_mionet_v2/training/audit.py`: zero baseline, parameter-gradient, and loss-dominance gates.
- `grouped_ufno_mionet_v2/training/trainer.py`: joint dual-head train/validation steps.
- `grouped_ufno_mionet_v2/training/checkpoint.py`: atomic `best.pt`/`last.pt` with scale and dataset binding.
- `scripts/fit_grouped_v2_normalization.py`: streaming train-only statistics.
- `scripts/build_grouped_v2_cache.py`: parallel structured train/validation cache builder.
- `scripts/overfit_grouped_v2.py`: mandatory eight-record gate.
- `scripts/train_grouped_v2.py`: bounded pilot and production entry point.
- `scripts/evaluate_grouped_v2.py`: full-field/receiver metrics and figures.
- `configs/grouped_v2/{smoke,overfit,pilot,production}.yaml`: immutable run recipes.
- `tests/grouped_ufno_mionet_v2/`: focused unit and integration tests.

### Task 1: Establish the V2 numerical contract

**Files:**
- Create: `grouped_ufno_mionet_v2/__init__.py`
- Create: `grouped_ufno_mionet_v2/config.py`
- Create: `grouped_ufno_mionet_v2/normalization.py`
- Create: `tests/grouped_ufno_mionet_v2/test_normalization.py`

- [ ] **Step 1: Write failing round-trip and split-binding tests**

```python
import pytest
import torch
from grouped_ufno_mionet_v2.normalization import PhysicalNormalizer, ScaleMetadata

def test_pressure_round_trip_is_order_one_and_fp32():
    meta = ScaleMetadata(velocity_center_mps=2500.0, velocity_scale_mps=1000.0,
                         pressure_scale_pa=1.0e-8, source_scales=(2000., 2000., 50., 1.2, 1.),
                         train_manifest_sha256="abc")
    norm = PhysicalNormalizer(meta)
    pressure = torch.tensor([1.0e-9, -2.0e-8])
    encoded = norm.encode_pressure(pressure, torch.tensor(2.0))
    assert encoded.dtype == torch.float32
    assert encoded.abs().max() >= 0.5
    torch.testing.assert_close(norm.decode_pressure(encoded, torch.tensor(2.0)), pressure)

def test_normalizer_rejects_wrong_manifest():
    with pytest.raises(ValueError, match="manifest"):
        PhysicalNormalizer.from_dict({"train_manifest_sha256": "a"}, expected_manifest="b")
```

- [ ] **Step 2: Verify RED**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_normalization.py`

Expected: FAIL because `grouped_ufno_mionet_v2` does not exist.

- [ ] **Step 3: Implement strict config and scale metadata**

```python
@dataclass(frozen=True)
class ScaleMetadata:
    velocity_center_mps: float
    velocity_scale_mps: float
    pressure_scale_pa: float
    source_scales: tuple[float, float, float, float, float]
    train_manifest_sha256: str

class PhysicalNormalizer:
    def __init__(self, metadata: ScaleMetadata):
        if metadata.velocity_scale_mps <= 0 or metadata.pressure_scale_pa <= 0:
            raise ValueError("normalization scales must be positive")
        self.metadata = metadata
    def encode_velocity(self, value):
        return (torch.as_tensor(value, dtype=torch.float32) - self.metadata.velocity_center_mps) / self.metadata.velocity_scale_mps
    def encode_source(self, value):
        scales = torch.tensor(self.metadata.source_scales, dtype=torch.float32, device=torch.as_tensor(value).device)
        return torch.as_tensor(value, dtype=torch.float32) / scales
    def encode_pressure(self, value, amplitude):
        return torch.as_tensor(value, dtype=torch.float32) / (self.metadata.pressure_scale_pa * torch.as_tensor(amplitude, dtype=torch.float32))
    def decode_pressure(self, value, amplitude):
        return torch.as_tensor(value, dtype=torch.float32) * self.metadata.pressure_scale_pa * torch.as_tensor(amplitude, dtype=torch.float32)
```

`V2Config.from_yaml` must reject unknown keys and require `dense_time_block`, `uniform_floor>=0.2`,
`pressure_percentile`, separate train/validation cache paths, and positive loss weights.

- [ ] **Step 4: Verify GREEN and commit**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_normalization.py`

Expected: PASS.

```bash
git add grouped_ufno_mionet_v2 tests/grouped_ufno_mionet_v2/test_normalization.py
git commit -m "feat: add v2 physical normalization contract"
```

### Task 2: Fit train-only statistics and build the structured cache

**Files:**
- Create: `scripts/fit_grouped_v2_normalization.py`
- Create: `scripts/build_grouped_v2_cache.py`
- Create: `grouped_ufno_mionet_v2/data/cache.py`
- Create: `tests/grouped_ufno_mionet_v2/test_cache.py`

- [ ] **Step 1: Write a failing tiny-cache schema test**

```python
def test_cache_exposes_structured_frames_traces_and_probabilities(tiny_grouped_v2_h5):
    ds = StructuredCacheDataset(tiny_grouped_v2_h5, expected_split="train")
    item = ds[0]
    assert item.dense_target.shape == (16, 9, 9)
    assert item.receiver_target.shape == (8, 21)
    assert item.query_coords.shape[-1] == 3
    assert item.source_map.shape == (1, 9, 9)
    assert item.sample_probability.min() > 0
    assert item.metadata["split"] == "train"
```

The fixture writes two records and exact datasets: `velocity_mps`, `source_parameters`,
`source_map`, `dense_time_indices`, `dense_target`, `receiver_zx_indices`,
`receiver_target`, `query_coords`, `query_target`, `sample_probability`, `sample_id`,
`group_id`, and `split`.

- [ ] **Step 2: Verify RED**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_cache.py`

Expected: FAIL with missing `StructuredCacheDataset`.

- [ ] **Step 3: Implement schema validation and the statistics CLI**

The statistics CLI must stream only rows whose HDF5 `split` equals `train`; compute velocity
median/IQR and a deterministic histogram approximation to the configured nonzero absolute-pressure
percentile; then atomically write JSON containing dataset SHA256, split count, algorithm, percentile,
and all scales. It must reject `validation`, `test_id`, and `ood_canonical` as fitting splits.

`StructuredCacheDataset` validates every dimension and digest before exposing tensors. It rejects
a cache whose `split` attribute differs from `expected_split`.

- [ ] **Step 4: Implement deterministic V2 cache selection**

```python
def select_dense_times(time_s, source_t0_s, count=16):
    k0 = int(np.searchsorted(time_s, source_t0_s, side="left"))
    anchors = np.array([k0, min(k0 + 1, len(time_s)-1)], dtype=np.int64)
    future = np.rint(np.linspace(min(k0 + 2, len(time_s)-1), len(time_s)-1, count-2)).astype(np.int64)
    return np.unique(np.concatenate((anchors, future)))[:count]
```

For each record, store 16 complete structured frames, 8 fixed receivers across all 401 saved times,
and 8192 query points. Query sampling uses 20% uniform points and 80% wave-energy points; save the
actual mixture probability for inverse-probability weighting. Build train and validation caches
separately via temporary files and atomic rename.

- [ ] **Step 5: Verify cache tests and commit**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_cache.py`

Expected: PASS, including split rejection and interrupted-cache rejection.

```bash
git add scripts/fit_grouped_v2_normalization.py scripts/build_grouped_v2_cache.py grouped_ufno_mionet_v2/data tests/grouped_ufno_mionet_v2/test_cache.py
git commit -m "feat: add structured dual-head v2 cache"
```

### Task 3: Preserve grouped one-source batching and pass source maps

**Files:**
- Create: `grouped_ufno_mionet_v2/data/batch.py`
- Create: `tests/grouped_ufno_mionet_v2/test_batch.py`

- [ ] **Step 1: Write failing grouping/source-isolation tests**

```python
def test_grouped_batch_deduplicates_medium_but_keeps_source_targets(records_same_medium):
    batch = pack_v2_groups(records_same_medium)
    assert batch.velocity_mps.shape[0] == 1
    assert batch.source_parameters.shape[0] == 2
    assert batch.source_map.shape == (2, 1, 9, 9)
    assert batch.dense_target.shape[0] == 2
    assert not torch.equal(batch.dense_target[0], batch.dense_target[1])
```

- [ ] **Step 2: Verify RED**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_batch.py`

Expected: FAIL with missing `pack_v2_groups`.

- [ ] **Step 3: Implement `V2MacroBatch` and packer**

The frozen dataclass carries unique velocity, record-to-medium mapping, source parameters/maps,
structured dense frames and indices, receiver traces/indices, query coordinates/targets/probabilities,
sample/group/family IDs. `pack_v2_groups` verifies byte-identical velocities within a group and
unit-mass nonnegative source maps. It never sums source maps or targets.

- [ ] **Step 4: Verify GREEN and commit**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_batch.py`

Expected: PASS.

```bash
git add grouped_ufno_mionet_v2/data/batch.py tests/grouped_ufno_mionet_v2/test_batch.py
git commit -m "feat: add v2 grouped source-isolated batches"
```

### Task 4: Implement the shared encoder and continuous query head

**Files:**
- Create: `grouped_ufno_mionet_v2/model/medium.py`
- Create: `grouped_ufno_mionet_v2/model/source.py`
- Create: `grouped_ufno_mionet_v2/model/query.py`
- Create: `tests/grouped_ufno_mionet_v2/test_query_head.py`

- [ ] **Step 1: Write failing normalized-output and source-map-gradient tests**

```python
def test_query_head_outputs_order_one_and_source_map_has_gradient(v2_model, v2_inputs):
    pred = v2_model.query_normalized(**v2_inputs)
    assert pred.shape == (2, 64)
    pred.square().mean().backward()
    grad = v2_model.source_encoder.map_proj[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

def test_query_chunking_is_equivalent(v2_model, v2_inputs):
    full = v2_model.query_normalized(**v2_inputs)
    chunked = v2_model.query_normalized(**v2_inputs, chunk_size=17)
    torch.testing.assert_close(chunked, full, rtol=2e-5, atol=2e-5)
```

- [ ] **Step 2: Verify RED**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_query_head.py`

Expected: FAIL because the V2 model modules are absent.

- [ ] **Step 3: Implement encoders and query head**

The medium encoder accepts already normalized velocity and returns four local pyramid levels plus
8×8 tokens. The source encoder combines normalized source parameters, bilinearly sampled local
features, and a CNN over the full unit-mass source map; no adaptive 4×4 pooling may erase sub-grid
location. The query head uses normalized coordinates, MIONet rank product, token attention, and
sampled full-resolution local features. It returns normalized pressure and does not multiply
`p_scale` internally.

- [ ] **Step 4: Verify GREEN and commit**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_query_head.py`

Expected: PASS.

```bash
git add grouped_ufno_mionet_v2/model tests/grouped_ufno_mionet_v2/test_query_head.py
git commit -m "feat: add v2 shared encoder and query head"
```

### Task 5: Implement the time-conditioned dense decoder

**Files:**
- Create: `grouped_ufno_mionet_v2/model/dense.py`
- Create: `grouped_ufno_mionet_v2/model/operator.py`
- Create: `tests/grouped_ufno_mionet_v2/test_dense_head.py`

- [ ] **Step 1: Write failing moving-wave and dual-head contract tests**

```python
def test_dense_decoder_outputs_requested_time_block(v2_model, v2_inputs):
    times = torch.linspace(.1, .3, 8).repeat(2, 1)
    dense = v2_model.dense_normalized(**v2_inputs, time_s=times)
    assert dense.shape == (2, 8, 201, 201)
    assert not torch.equal(dense[:, 0], dense[:, -1])

def test_dense_parameters_receive_gradient(v2_model, v2_inputs):
    v2_model.dense_normalized(**v2_inputs, time_s=torch.tensor([[.1, .2], [.1, .2]])).mean().backward()
    assert all(p.grad is not None for p in v2_model.dense_decoder.parameters() if p.requires_grad)
```

- [ ] **Step 2: Verify RED**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_dense_head.py`

Expected: FAIL with missing dense decoder.

- [ ] **Step 3: Implement time-conditioned multiscale decoding**

Fuse upsampled pyramid levels once per source record. Embed each physical time with Fourier features;
generate per-channel FiLM scale/bias; expand only the fused feature across a bounded time block;
apply two depthwise-separable residual blocks and a FP32 output projection. Inject a projected
source map at full resolution. `dense_normalized` accepts at most the configured block size and
`predict_wavefield` streams longer time arrays block by block.

- [ ] **Step 4: Add physical-unit public APIs**

```python
def query_pressure(self, velocity_mps, source, source_map, coords, normalizer, chunk_size=None):
    normalized = self.query_normalized(velocity_mps, source, source_map, coords, normalizer, chunk_size)
    return normalizer.decode_pressure(normalized.float(), source[:, 4:5])

def predict_wavefield(self, velocity_mps, source, source_map, time_s, normalizer):
    blocks = [self.dense_normalized(..., time_s=t) for t in time_s.split(self.config.dense_time_block, dim=1)]
    normalized = torch.cat(blocks, dim=1)
    return normalizer.decode_pressure(normalized.float(), source[:, 4, None, None, None])
```

- [ ] **Step 5: Verify GREEN and commit**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_dense_head.py tests/grouped_ufno_mionet_v2/test_query_head.py`

Expected: PASS and trainable dense parameters all have gradients.

```bash
git add grouped_ufno_mionet_v2/model tests/grouped_ufno_mionet_v2/test_dense_head.py
git commit -m "feat: add time-conditioned v2 dense decoder"
```

### Task 6: Implement coherent normalized losses

**Files:**
- Create: `grouped_ufno_mionet_v2/losses.py`
- Create: `tests/grouped_ufno_mionet_v2/test_losses.py`

- [ ] **Step 1: Write failing scale and axis tests**

```python
def test_zero_baseline_and_perfect_prediction_are_distinct():
    target = torch.randn(2, 8, 9, 9)
    perfect = dual_head_losses(target, target, target[:, :, :1, :], target[:, :, :1, :])
    zero = dual_head_losses(torch.zeros_like(target), target,
                            torch.zeros_like(target[:, :, :1, :]), target[:, :, :1, :])
    assert perfect.total == 0
    assert zero.total > 0

def test_spectral_losses_reject_flat_random_queries():
    with pytest.raises(ValueError, match="structured"):
        spatial_fft_loss(torch.randn(2, 1024), torch.randn(2, 1024))
```

- [ ] **Step 2: Verify RED**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_losses.py`

Expected: FAIL with missing losses.

- [ ] **Step 3: Implement losses**

Use per-record relative L2 with active-energy epsilon, SmoothL1 in normalized units, finite-difference
spatial gradients on `[B,T,Z,X]`, `rfft2` only on structured frames, and `rfft` only on
`[B,receiver,time]` traces. Query loss uses saved inverse-probability weights normalized per
record. Consistency samples the dense grid and supervises both heads against target before applying
a symmetric agreement term.

- [ ] **Step 4: Verify GREEN and commit**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_losses.py`

Expected: PASS; perfect loss is zero and zero baseline is finite/nonzero.

```bash
git add grouped_ufno_mionet_v2/losses.py tests/grouped_ufno_mionet_v2/test_losses.py
git commit -m "feat: add coherent normalized dual-head losses"
```

### Task 7: Build audited joint training, validation, and checkpoints

**Files:**
- Create: `grouped_ufno_mionet_v2/training/audit.py`
- Create: `grouped_ufno_mionet_v2/training/checkpoint.py`
- Create: `grouped_ufno_mionet_v2/training/trainer.py`
- Create: `tests/grouped_ufno_mionet_v2/test_trainer.py`

- [ ] **Step 1: Write failing missing-gradient and best-checkpoint tests**

```python
def test_gradient_audit_rejects_unused_dense_head(v2_model):
    with pytest.raises(RuntimeError, match="dense_decoder"):
        require_gradients(v2_model, required_prefixes=("medium_encoder", "source_encoder", "query_head", "dense_decoder"))

def test_best_checkpoint_requires_validation_improvement(tmp_path, trainer):
    trainer.save_epoch(tmp_path, validation_score=.4)
    trainer.save_epoch(tmp_path, validation_score=.6)
    assert load_state(tmp_path/"best.pt")["validation_score"] == .4
```

- [ ] **Step 2: Verify RED**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_trainer.py`

Expected: FAIL with missing training modules.

- [ ] **Step 3: Implement one joint step**

The trainer passes `source_map` unconditionally, computes dense/query/trace/consistency predictions,
uses FP32 losses, accumulates bounded time blocks, and updates once per macro-batch. AMP is BF16 on
supported CUDA; FFT and output projections opt out of autocast. Every result logs every unweighted
and weighted loss component, both relative L2 metrics, gradient norms, throughput, and memory.

- [ ] **Step 4: Implement gates and atomic checkpointing**

`require_gradients` lists every missing/nonfinite required parameter and stops immediately.
`LossDominanceMonitor` stops after an auxiliary/data ratio above 10 for five consecutive steps.
Atomic checkpoints contain model/optimizer/scaler/scheduler, epoch/step, sampler state, normalizer,
dataset/cache digests, config digest, zero baseline, validation metrics, and RNG states.
`last.pt` saves every epoch; `best.pt` changes only on lower composite validation score.

- [ ] **Step 5: Verify GREEN and commit**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_trainer.py`

Expected: PASS, including resume equality and deliberate unused-head failure.

```bash
git add grouped_ufno_mionet_v2/training tests/grouped_ufno_mionet_v2/test_trainer.py
git commit -m "feat: add audited v2 dual-head trainer"
```

### Task 8: Enforce the eight-record overfit gate

**Files:**
- Create: `scripts/overfit_grouped_v2.py`
- Create: `configs/grouped_v2/smoke.yaml`
- Create: `configs/grouped_v2/overfit.yaml`
- Create: `tests/grouped_ufno_mionet_v2/test_overfit_gate.py`

- [ ] **Step 1: Write a failing gate-decision test**

```python
def test_gate_requires_both_heads_to_beat_zero():
    result = evaluate_overfit_gate(query_rel_l2=.08, dense_rel_l2=.07,
                                   zero_query_rel_l2=1., zero_dense_rel_l2=1.,
                                   missing_gradients=[])
    assert result.passed
    assert not evaluate_overfit_gate(.08, 1.2, 1., 1., []).passed
    assert not evaluate_overfit_gate(.08, .07, 1., 1., ["source_encoder.map_proj.weight"]).passed
```

- [ ] **Step 2: Verify RED**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_overfit_gate.py`

Expected: FAIL with missing gate API.

- [ ] **Step 3: Implement the gate CLI**

Select eight deterministic train records with nonzero onset/early/middle/late energy and at least
two medium families. Train until both relative L2 metrics are below 0.10 or the configured maximum
step is reached. Save zero/current metrics, gradient audit, loss history, `gate.json`, checkpoint,
and true/prediction/error plus receiver figures. Exit nonzero on failure.

- [ ] **Step 4: Run CPU tiny smoke then CUDA eight-record gate**

Run: `/home/jiayh/miniconda3/bin/python scripts/overfit_grouped_v2.py --config configs/grouped_v2/smoke.yaml --device cpu`

Expected: tensor/gradient/checkpoint smoke exits 0.

Run: `CUDA_VISIBLE_DEVICES=0 /home/jiayh/miniconda3/bin/python scripts/overfit_grouped_v2.py --config configs/grouped_v2/overfit.yaml --device cuda`

Expected: `gate.json` has `"passed": true`, both relative L2 values below 0.10, and no missing gradients.

- [ ] **Step 5: Commit**

```bash
git add scripts/overfit_grouped_v2.py configs/grouped_v2 tests/grouped_ufno_mionet_v2/test_overfit_gate.py
git commit -m "feat: enforce v2 dual-head overfit gate"
```

### Task 9: Add the bounded pilot, evaluation, and figures

**Files:**
- Create: `scripts/train_grouped_v2.py`
- Create: `scripts/evaluate_grouped_v2.py`
- Create: `configs/grouped_v2/pilot.yaml`
- Create: `configs/grouped_v2/production.yaml`
- Create: `tests/grouped_ufno_mionet_v2/test_cli.py`

- [ ] **Step 1: Write failing launch-guard tests**

```python
def test_production_refuses_missing_or_failed_overfit_gate(tmp_path):
    with pytest.raises(RuntimeError, match="overfit gate"):
        validate_launch_gate(tmp_path/"missing.json")
    (tmp_path/"failed.json").write_text('{"passed": false}')
    with pytest.raises(RuntimeError, match="overfit gate"):
        validate_launch_gate(tmp_path/"failed.json")
```

- [ ] **Step 2: Verify RED**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_cli.py`

Expected: FAIL with missing launch guard.

- [ ] **Step 3: Implement pilot/production CLI**

The CLI requires an approved `gate.json`, distinct train/validation caches, and fresh V2 output
directory. The pilot runs a bounded configured step count, validates every epoch, and must improve
both dense and query composite validation scores over zero. Only a passed pilot may set
`production_authorized=true`; production resumes from the V2 pilot best checkpoint, never V1.

- [ ] **Step 4: Implement evaluator**

For `validation`, `test_id`, and individual `ood_canonical` records, compute full-field and
receiver metrics, per-time P50/P95, energy, spatial/trace spectrum, throughput, and zero baseline.
Generate colorblind-safe true/prediction/error snapshots with honest row-specific scales plus a
shared normalized-error panel, and receiver gather/trace plots. Save raw NPZ/HDF5 predictions and
a JSON report so figures do not require rerunning inference.

- [ ] **Step 5: Verify CLI tests and commit**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2/test_cli.py`

Expected: PASS; failed gate/pilot cannot launch production.

```bash
git add scripts/train_grouped_v2.py scripts/evaluate_grouped_v2.py configs/grouped_v2 tests/grouped_ufno_mionet_v2/test_cli.py
git commit -m "feat: add gated v2 pilot and evaluation"
```

### Task 10: Build real caches, run the pilot, and launch production only on evidence

**Files:**
- Create: `artifacts/grouped_ufno_mionet_v2/launch_report.md`

- [ ] **Step 1: Run the complete focused suite**

Run: `/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v2 tests/grouped_ufno_mionet`

Expected: all tests PASS.

- [ ] **Step 2: Fit statistics and build isolated caches**

Run:

```bash
/home/jiayh/miniconda3/bin/python scripts/fit_grouped_v2_normalization.py \
  --dataset /home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5 \
  --output /home/jiayh/Data/data/processed/grouped_v2_normalization.json
/home/jiayh/miniconda3/bin/python scripts/build_grouped_v2_cache.py --split train --workers 8
/home/jiayh/miniconda3/bin/python scripts/build_grouped_v2_cache.py --split validation --workers 8
```

Expected: atomic train/validation HDF5 caches pass schema/digest checks; no test/OOD rows appear.

- [ ] **Step 3: Run overfit gate and bounded pilot**

Run the Task 8 CUDA gate, then:

```bash
CUDA_VISIBLE_DEVICES=0 /home/jiayh/miniconda3/bin/python scripts/train_grouped_v2.py \
  --config configs/grouped_v2/pilot.yaml \
  --gate artifacts/grouped_ufno_mionet_v2/overfit/gate.json \
  --output artifacts/grouped_ufno_mionet_v2/pilot
```

Expected: pilot improves both heads over zero and writes `pilot_gate.json` with
`production_authorized=true`.

- [ ] **Step 4: Launch production under nohup only if the pilot passes**

```bash
nohup /home/jiayh/miniconda3/bin/python -u scripts/train_grouped_v2.py \
  --config configs/grouped_v2/production.yaml \
  --gate artifacts/grouped_ufno_mionet_v2/pilot/pilot_gate.json \
  --resume artifacts/grouped_ufno_mionet_v2/pilot/best.pt \
  --output artifacts/grouped_ufno_mionet_v2/production \
  > artifacts/grouped_ufno_mionet_v2/production/train.nohup.log 2>&1 &
```

Expected: PID remains alive after the first validation boundary, both heads have gradients, GPU
utilization is measured, and `best.pt`/`last.pt` are independently maintained.

- [ ] **Step 5: Write the launch report and commit metadata**

Record exact cache/config/checkpoint hashes, gate results, test output, PID/log paths, first
validation metrics, GPU utilization/memory/power, and any failed accuracy gate. Do not claim
accuracy or speed before measurement.

```bash
git add artifacts/grouped_ufno_mionet_v2/launch_report.md
git commit -m "docs: record grouped v2 gated launch evidence"
```

## Self-review

- Spec coverage: Tasks 1–2 implement numerical and data contracts; Tasks 3–5 implement one-source
  grouped dual heads; Task 6 implements physically coherent losses; Tasks 7–9 implement audits,
  validation, best checkpoints, figures, and launch guards; Task 10 supplies measured real-data
  evidence and production launch.
- FWI scope: no inversion, velocity-gradient gate, acquisition design, FWI loss, or FWI benchmark is
  implemented. Public interfaces avoid intentionally blocking a future separate FWI design.
- Placeholder scan: every task identifies exact paths, APIs, commands, expected failures, and pass
  evidence; no unresolved implementation placeholders remain.
- Type consistency: pressure is normalized inside training APIs and restored only in public physical
  APIs; dense fields use `[B,T,Z,X]`; receiver traces use `[B,R,T]`; query coordinates use
  `[B,Q,3]` in `[x,z,t]` order throughout.

