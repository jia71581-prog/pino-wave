# Saved-Time V5 Training-Contract Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a trustworthy V4-model baseline that covers all 2,240 training records per epoch, covers at least 120 exact stored times per record by epoch 30, progressively unfreezes the full model, and evaluates representative held-out support.

**Architecture:** Keep the existing `SavedTimePhaseOperatorV4` unchanged so this experiment isolates the training contract. Add deterministic full-support schedule and time-policy modules, wire them into the exact-frame dataset, add staged AdamW parameter control and streamed metrics, then use a dedicated resumable runner and production config.

**Tech Stack:** Python 3.12, PyTorch 2.7, NumPy, HDF5/h5py, PyYAML, pytest, CUDA/RTX 3090.

---

## File map

- Create `saved_time_phase_operator_v4/full_support.py`: epoch schedule, coverage audit, staged parameter policy, warmup/cosine learning-rate factors.
- Modify `saved_time_phase_operator_v4/sampling.py`: deterministic exact-time appearance policy and coverage ledger helpers.
- Modify `saved_time_phase_operator_v4/data.py`: materialize appearance-aware 4-frame training and 16-frame validation requests.
- Create `saved_time_phase_operator_v4/streaming_metrics.py`: bounded-memory metrics over arbitrary exact-time blocks.
- Create `scripts/train_saved_time_v4_full_support.py`: identity-bound, resumable, epoch-aware training and validation runner.
- Create `scripts/audit_saved_time_v4_full_support.py`: production-manifest schedule/time audit without GPU allocation.
- Create `configs/saved_time_v4/full_support_adamw_batch48.yaml`: registered production settings.
- Create focused tests under `tests/saved_time_phase_operator_v4/`.

### Task 1: Deterministic all-record epoch schedule

**Files:**
- Create: `saved_time_phase_operator_v4/full_support.py`
- Create: `tests/saved_time_phase_operator_v4/test_full_support.py`

- [ ] **Step 1: Write failing schedule tests**

```python
from collections import Counter

from saved_time_phase_operator_v4.full_support import (
    audit_epoch_schedule,
    build_full_support_schedule,
    schedule_digest,
)


def test_full_support_epoch_covers_2240_records_with_batch48_padding():
    schedule = build_full_support_schedule(
        record_count=2240, epochs=2, macro_records=12,
        macros_per_update=4, seed=307,
    )
    assert len(schedule) == 2 * 188
    audit = audit_epoch_schedule(schedule, epoch=0, record_count=2240)
    assert audit.record_count == 2240
    assert audit.appearances == 2256
    assert audit.minimum_appearances == 1
    assert audit.maximum_appearances == 2
    assert audit.optimizer_updates == 47


def test_full_support_schedule_is_deterministic_and_epoch_distinct():
    first = build_full_support_schedule(48, epochs=3, macro_records=12, macros_per_update=4, seed=11)
    repeated = build_full_support_schedule(48, epochs=3, macro_records=12, macros_per_update=4, seed=11)
    changed = build_full_support_schedule(48, epochs=3, macro_records=12, macros_per_update=4, seed=12)
    assert first == repeated
    assert first != changed
    assert schedule_digest(first) == schedule_digest(repeated)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_full_support.py`

Expected: FAIL with `ModuleNotFoundError: saved_time_phase_operator_v4.full_support`.

- [ ] **Step 3: Implement the minimal schedule API**

```python
@dataclass(frozen=True)
class FullSupportStepSpec:
    step: int
    epoch: int
    record_indices: tuple[int, ...]
    appearance_indices: tuple[int, ...]


def build_full_support_schedule(record_count, *, epochs, macro_records, macros_per_update, seed):
    update_records = macro_records * macros_per_update
    padded = math.ceil(record_count / update_records) * update_records
    appearances = np.zeros(record_count, dtype=np.int64)
    result = []
    step = 0
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch * 104729).permutation(record_count).tolist()
        order.extend(order[: padded - record_count])
        for start in range(0, padded, macro_records):
            selected = tuple(int(value) for value in order[start : start + macro_records])
            current = tuple(int(appearances[value]) for value in selected)
            for value in selected:
                appearances[value] += 1
            result.append(FullSupportStepSpec(step, epoch, selected, current))
            step += 1
    return tuple(result)
```

Also implement `EpochScheduleAudit`, strict range/balance/update-alignment checks, and SHA-256 over canonical step payloads.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_full_support.py`

Expected: all schedule tests PASS.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4/full_support.py tests/saved_time_phase_operator_v4/test_full_support.py
git commit -m "feat(v5): add deterministic full-support schedule"
```

### Task 2: Exact-time coverage policy and ledger

**Files:**
- Modify: `saved_time_phase_operator_v4/sampling.py`
- Modify: `tests/saved_time_phase_operator_v4/test_sampling.py`

- [ ] **Step 1: Write failing appearance-policy tests**

```python
def test_appearance_sampler_reaches_120_unique_times_at_epoch_30():
    axis = np.linspace(0.0, 1.0, 401)
    seen = set()
    pre = 0
    for appearance in range(30):
        indices = appearance_time_indices(
            axis, source_t0_s=0.05, sample_id="sample-1",
            appearance=appearance, seed=307, count=4,
        )
        seen.update(indices.tolist())
        pre += int((indices < 20).sum())
    assert len(seen) >= 120
    assert pre / (30 * 4) <= 0.05


def test_appearance_sampler_is_exact_deterministic_and_source_specific():
    axis = np.linspace(0.0, 1.0, 401)
    one = appearance_time_indices(axis, source_t0_s=0.1, sample_id="a", appearance=4, seed=9, count=4)
    two = appearance_time_indices(axis, source_t0_s=0.1, sample_id="a", appearance=4, seed=9, count=4)
    other = appearance_time_indices(axis, source_t0_s=0.1, sample_id="b", appearance=4, seed=9, count=4)
    np.testing.assert_array_equal(one, two)
    assert not np.array_equal(one, other)
    assert len(set(one.tolist())) == 4
```

- [ ] **Step 2: Run and verify RED**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_sampling.py`

Expected: FAIL because `appearance_time_indices` does not exist.

- [ ] **Step 3: Implement deterministic bin permutations**

Use SHA-256 of `seed:sample_id:bin_name` as the NumPy RNG seed. Split active indices into early, middle, and late thirds. Every fifth appearance uses one pre-onset frame; other appearances use a second early-active frame. Select early positions `2*appearance` and `2*appearance+1`, and middle/late position `appearance`, all through deterministic permutations. Reject `count != 4` in the training API.

```python
def appearance_time_indices(time_s, *, source_t0_s, sample_id, appearance, seed, count=4):
    axis = _validated_axis(time_s)
    onset = int(np.searchsorted(axis, source_t0_s, side="left"))
    early, middle, late = np.array_split(np.arange(onset, len(axis)), 3)
    early_order = _permuted(early, seed, sample_id, "early")
    selected = [int(early_order[(2 * appearance) % len(early_order)])]
    if appearance % 5 == 0:
        pre_order = _permuted(np.arange(0, onset), seed, sample_id, "pre")
        selected.insert(0, int(pre_order[(appearance // 5) % len(pre_order)]))
    else:
        selected.insert(0, int(early_order[(2 * appearance + 1) % len(early_order)]))
    selected.extend((
        int(_permuted(middle, seed, sample_id, "middle")[appearance % len(middle)]),
        int(_permuted(late, seed, sample_id, "late")[appearance % len(late)]),
    ))
    return np.asarray(selected, dtype=np.int64)
```

Add `coverage_ledger(schedule, time_s, source_t0_by_record, sample_id_by_record, seed)` returning per-record bit-packed 401-index masks and min/median/max counts.

- [ ] **Step 4: Run sampling tests and coverage audit**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_sampling.py`

Expected: all sampling tests PASS and minimum unique coverage is at least 120 after 30 appearances.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4/sampling.py tests/saved_time_phase_operator_v4/test_sampling.py
git commit -m "feat(v5): guarantee exact-time coverage by appearance"
```

### Task 3: Wire appearance-aware training and 16-frame validation into the dataset

**Files:**
- Modify: `saved_time_phase_operator_v4/data.py`
- Modify: `tests/saved_time_phase_operator_v4/test_data.py`

- [ ] **Step 1: Write failing dataset tests**

Add tests using the production fixture when available:

```python
def test_full_support_spec_uses_appearance_indices(exact_batch):
    _, manifest = exact_batch
    spec = FullSupportStepSpec(step=0, epoch=0, record_indices=tuple(range(12)), appearance_indices=(7,) * 12)
    dataset = ExactStoredTimeBatchDataset(
        SOURCE_H5, manifest, split="validation", schedule=(spec,),
        query_points=1, seed=307, time_policy="appearance4",
    )
    batch = dataset[0]
    assert batch.requested_time_s.shape == (12, 4)
    assert batch.target_exact.all()


def test_validation_policy_materializes_sixteen_exact_frames(exact_batch):
    _, manifest = exact_batch
    dataset = ExactStoredTimeBatchDataset(
        SOURCE_H5, manifest, split="validation", schedule=(spec,),
        query_points=1, seed=307, time_policy="validation16",
    )
    batch = dataset[0]
    assert batch.requested_time_s.shape == (12, 16)
    assert torch.equal(batch.left_index, batch.right_index)
```

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because `time_policy` and `appearance_indices` are not consumed.

- [ ] **Step 3: Implement policy dispatch without breaking V4 callers**

Keep `time_policy="legacy4"` as the default. For `appearance4`, require `appearance_indices` and call `appearance_time_indices`. For `validation16`, select one deterministic pre-onset index and five indices from each active third. All policies pass stored axis values directly to `read_wavefield`; preserve the `exact` and `left_index == right_index` guards.

- [ ] **Step 4: Run data and existing V4 regression tests**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_data.py tests/saved_time_phase_operator_v4/test_sampling.py`

Expected: PASS, including unchanged legacy four-frame tests.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4/data.py tests/saved_time_phase_operator_v4/test_data.py
git commit -m "feat(v5): materialize appearance-aware exact frames"
```

### Task 4: Staged unfreezing and AdamW policy

**Files:**
- Modify: `saved_time_phase_operator_v4/full_support.py`
- Modify: `tests/saved_time_phase_operator_v4/test_full_support.py`

- [ ] **Step 1: Write failing stage-policy tests**

```python
def test_trainable_stages_expand_without_overlap(tiny_v4_model):
    stage1 = configure_trainable_stage(tiny_v4_model, epoch=1)
    assert stage1.trainable_prefixes == ("dense_decoder", "source_encoder", "fusion")
    stage3 = configure_trainable_stage(tiny_v4_model, epoch=3)
    assert "coordinate_encoder" in stage3.trainable_prefixes
    assert "travel_branch" in stage3.trainable_prefixes
    stage6 = configure_trainable_stage(tiny_v4_model, epoch=6)
    assert stage6.trainable_parameters == sum(p.numel() for p in tiny_v4_model.parameters())


def test_optimizer_groups_are_exclusive_and_exhaustive(tiny_v4_model):
    optimizer = build_staged_adamw(tiny_v4_model, dense_lr=2e-4, geometry_lr=1e-4, backbone_lr=2e-5, weight_decay=1e-6)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in tiny_v4_model.parameters()}
```

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because stage functions are absent.

- [ ] **Step 3: Implement named-prefix stages and optimizer groups**

Define groups `dense=(dense_decoder,source_encoder,fusion)`, `geometry=(coordinate_encoder,travel_branch)`, and `backbone=(medium_encoder)`. Build the optimizer once with all groups; `configure_trainable_stage` changes only `requires_grad`. Implement three-epoch linear warmup followed by cosine decay to a factor of 0.05.

- [ ] **Step 4: Run and verify GREEN**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_full_support.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4/full_support.py tests/saved_time_phase_operator_v4/test_full_support.py
git commit -m "feat(v5): add staged full-model AdamW policy"
```

### Task 5: Bounded-memory arbitrary-time validation metrics

**Files:**
- Create: `saved_time_phase_operator_v4/streaming_metrics.py`
- Create: `tests/saved_time_phase_operator_v4/test_streaming_metrics.py`

- [ ] **Step 1: Write failing equivalence and grouping tests**

```python
def test_streaming_metrics_match_single_batch_record_l2():
    target = torch.randn(6, 16, 9, 9)
    prediction = target + 0.1 * torch.randn_like(target)
    accumulator = ExactWavefieldMetricAccumulator(energy_floor_fraction=0.01)
    accumulator.update(prediction[:2], target[:2], families=("uniform",) * 2, group_ids=("u0", "u1"), sample_ids=("a", "b"), time_indices=torch.arange(16).repeat(2, 1))
    accumulator.update(prediction[2:], target[2:], families=("layered",) * 4, group_ids=("l0",) * 4, sample_ids=("c", "d", "e", "f"), time_indices=torch.arange(16).repeat(4, 1))
    result = accumulator.finalize()
    expected = ((prediction - target).flatten(1).norm(dim=1) / target.flatten(1).norm(dim=1)).mean()
    assert result["aggregate_relative_l2"] == pytest.approx(float(expected), rel=1e-6)
    assert result["record_count"] == 6
    assert result["unique_time_index_count"] == 16
    assert set(result["family_relative_l2"]) == {"uniform", "layered"}
    assert set(result["medium_relative_l2"]) == {"u0", "u1", "l0"}
```

- [ ] **Step 2: Run and verify RED**

Expected: FAIL with missing `streaming_metrics` module.

- [ ] **Step 3: Implement an accumulator using scalar sums**

Accumulate per-record squared error/reference norms, RMSE sums/counts, family/group/source sums, active-time-bin sums, phase dot products/norms, and three radial FFT-band norms. Reject duplicate `(sample_id,time_index)` pairs when `require_unique=True`. Keep tensors only for the current update and move scalar sums to CPU.

- [ ] **Step 4: Run focused tests**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_streaming_metrics.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add saved_time_phase_operator_v4/streaming_metrics.py tests/saved_time_phase_operator_v4/test_streaming_metrics.py
git commit -m "feat(v5): add streamed exact-time metrics"
```

### Task 6: Resumable full-support training runner

**Files:**
- Create: `scripts/train_saved_time_v4_full_support.py`
- Create: `tests/saved_time_phase_operator_v4/test_full_support_runner.py`

- [ ] **Step 1: Write failing runner-policy tests**

Test pure helpers without allocating CUDA:

```python
def test_epoch_ranges_align_four_macros_per_update():
    ranges = epoch_step_ranges(total_macros=376, macros_per_epoch=188)
    assert ranges == ((0, 188), (188, 376))


def test_validation_scope_rotates_and_expands_every_fifth_epoch():
    assert validation_scope(epoch=1, validation_records=480) == ("panel", 48)
    assert validation_scope(epoch=5, validation_records=480) == ("all_records", 480)


def test_three_epoch_gate_requires_best_nonincrease_and_family_safety():
    reports = [{"score": 0.56, "family": {"uniform": 0.55}}, {"score": 0.54, "family": {"uniform": 0.54}}, {"score": 0.53, "family": {"uniform": 0.53}}]
    assert pilot_gate(reports, family_tolerance=0.03)["passed"]
```

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement the runner**

The runner must:

1. Load the bound V4 parent and verify manifest/checkpoint/config identity.
2. Build the complete 50-epoch schedule before model allocation and write `schedule_audit.json`.
3. Use `appearance4`, merge four macros, and use stage-aware physical microbatches: 12 for dense/source/fusion, 8 with coordinate/travel, and 4 with the medium backbone. Normalize loss by the resulting number of microbatches so the effective batch remains 48.
4. Configure the epoch stage, record a live training GPU snapshot, perform AdamW plus warmup/cosine, and checkpoint every epoch.
5. Write a bit-packed `coverage_epoch_XXXX.npz` and verify minimum coverage at epoch 30.
6. Evaluate rotating 48-record panels every epoch and all 480 records every fifth epoch with `validation16`.
7. Select best checkpoints only from all-record evaluations.
8. Resume only from a complete epoch with matching run identity, optimizer state, schedule digest, and coverage ledger.
9. In pilot mode, evaluate the same fixed 48-record/16-time panel after each epoch so the three pilot scores are comparable.
10. After the selected production checkpoint is fixed, stream all 480 validation records over all 401 stored indices in bounded time blocks and write a separate sealed report before marking the run complete.

Expose `--config`, `--smoke-updates`, `--pilot`, and `--audit-only`. `--pilot` uses `gate.pilot_epochs`, writes under `artifact_dir/pilot`, and has an identity distinct from the production run. `--smoke-updates` must use the full production model but may truncate the data schedule after the audit has passed.

- [ ] **Step 4: Run runner helper tests**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4/test_full_support_runner.py`

Expected: PASS.

- [ ] **Step 5: Run all CPU-safe V4 tests**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4 -k 'not production and not cuda'`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_saved_time_v4_full_support.py tests/saved_time_phase_operator_v4/test_full_support_runner.py
git commit -m "feat(v5): add resumable full-support trainer"
```

### Task 7: Production audit, configuration, and CUDA smoke gate

**Files:**
- Create: `scripts/audit_saved_time_v4_full_support.py`
- Create: `configs/saved_time_v4/full_support_adamw_batch48.yaml`
- Modify: `tests/saved_time_phase_operator_v4/test_full_support_runner.py`

- [ ] **Step 1: Write failing config/audit tests**

Assert the YAML registers 50 epochs, 4 macros/update, macro size 12, microbatch 12, workers 8, prefetch 4, validation panel 48, all-record cadence 5, and final 401-frame evaluation.

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because config and audit CLI are absent.

- [ ] **Step 3: Add the registered config**

```yaml
base_config: configs/grouped_v3/continuous_pilot.yaml
parent_checkpoint: /data/jiayh/saved_time_v4_spectrum_ablation_batch24/variants/deep_phase_plain/best.pt
parent_identity: /data/jiayh/saved_time_v4_spectrum_ablation_batch24/variants/deep_phase_plain/run_identity.json
artifact_dir: /data/jiayh/saved_time_v5_training_recovery_batch48
seed: 307
epochs: 50
macro_records: 12
macros_per_update: 4
microbatch_records: 12
query_points: 1
workers: 8
prefetch_factor: 4
time_policy: appearance4
validation:
  panel_records: 48
  frames_per_record: 16
  all_records_every: 5
  final_frames_per_record: 401
optimizer:
  dense_learning_rate: 0.0002
  geometry_learning_rate: 0.0001
  backbone_learning_rate: 0.00002
  weight_decay: 0.000001
  warmup_epochs: 3
  minimum_factor: 0.05
  gradient_clip: 1.0
loss:
  spatial_gradient: 0.1
  spectrum: 0.2
gate:
  maximum_peak_cuda_gib: 23.0
  pilot_epochs: 3
  family_regression_tolerance: 0.03
  target_aggregate_relative_l2: 0.10
  target_family_relative_l2: 0.12
```

The audit CLI loads the manifest and source onset/sample IDs, builds the schedule, computes coverage through epochs 30 and 50, and exits nonzero unless all registered gates pass.

- [ ] **Step 4: Run the production audit**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python scripts/audit_saved_time_v4_full_support.py --config configs/saved_time_v4/full_support_adamw_batch48.yaml`

Expected: JSON reports 2,240/2,240 records per epoch, 47 updates/epoch, minimum time coverage at least 120 at epoch 30, no interpolated requests, and exit 0.

- [ ] **Step 5: Run one-update CUDA smoke after the current ASAM process releases the GPU**

Run: `/home/jiayh/miniforge3/envs/PINO/bin/python scripts/train_saved_time_v4_full_support.py --config configs/saved_time_v4/full_support_adamw_batch48.yaml --smoke-updates 1`

Expected: finite loss and gradients, training-step GPU snapshot, peak allocation below 23 GiB, atomic checkpoint, exit 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit_saved_time_v4_full_support.py configs/saved_time_v4/full_support_adamw_batch48.yaml tests/saved_time_phase_operator_v4/test_full_support_runner.py
git commit -m "feat(v5): register full-support production recovery"
```

### Task 8: Three-epoch pilot, long-run launch, and monitoring handoff

**Files:**
- Modify: `README.md`
- Create: `docs/superpowers/reports/2026-07-18-saved-time-v5-training-recovery-launch.md`

- [ ] **Step 1: Run the three-epoch pilot with nohup**

```bash
nohup /home/jiayh/miniforge3/envs/PINO/bin/python -u scripts/train_saved_time_v4_full_support.py \
  --config configs/saved_time_v4/full_support_adamw_batch48.yaml --pilot \
  > /data/jiayh/saved_time_v5_training_recovery_batch48/pilot_launcher.log 2>&1 &
```

Record the PID and command atomically in `pilot_launcher.json`.

- [ ] **Step 2: Verify all three pilot epochs**

Expected: each epoch covers 2,240 records, checkpoints exist, every scheduled trainable group has finite gradients, and the best held-out score is non-increasing without family regression over three percent.

- [ ] **Step 3: Launch or resume the 50-epoch production run**

Only if the pilot gate passes:

```bash
nohup /home/jiayh/miniforge3/envs/PINO/bin/python -u scripts/train_saved_time_v4_full_support.py \
  --config configs/saved_time_v4/full_support_adamw_batch48.yaml \
  > /data/jiayh/saved_time_v5_training_recovery_batch48/launcher.log 2>&1 &
```

- [ ] **Step 4: Verify live execution**

Check that the PID is alive, the checkpoint/log timestamps advance, GPU utilization is nonzero during training, and no `terminal.json` failure exists. Do not claim the 10% target before the streamed final evaluation finishes.

- [ ] **Step 5: Document the exact launch state and commit**

```bash
git add README.md docs/superpowers/reports/2026-07-18-saved-time-v5-training-recovery-launch.md
git commit -m "docs(v5): record full-support recovery launch"
```

## Final verification

- [ ] Run `/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest -q tests/saved_time_phase_operator_v4`.
- [ ] Run `git diff --check` and confirm only intended files are committed.
- [ ] Verify the active run identity binds the manifest digest, parent digest, schedule digest, and exact 401-value time-axis digest.
- [ ] Verify no receiver traces, interpolated targets, anomaly records, or FWI components enter the run.
- [ ] Verify the earlier ASAM process was not signaled or terminated by this implementation.
