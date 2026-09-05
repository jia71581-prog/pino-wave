# V5 Efficient Causal Instance Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leakage-proof, parameter-efficient instance adaptation path for the V5 stored-time operator using only the two early post-onset snapshots, then evaluate Uniform, Layered, and Marmousi instances with full-field and derived receiver diagnostics.

**Architecture:** Keep `SavedTimePhaseOperatorV4` as a frozen parent and wrap it with a zero-initialized onset conditioner plus a low-rank residual/FiLM adapter. A guarded dataset exposes only `k0/k1`; optional LWC-84 bridge targets are generated from those inputs and marked synthetic. A sealed evaluator reads later truth only after adaptation and acceptance are complete, streams all 401 saved times, and renders field/waveform comparisons.

**Tech Stack:** Python 3, PyTorch, HDF5/h5py, NumPy, Matplotlib, existing `saved_time_phase_operator_v4`, `grouped_ufno_mionet_v3` manifest/normalizer/checkpoint utilities, and `src/fno_acoustic.data_generation.solver_lwc84.LWC84CPMLSolver`.

---

## Task 1: Establish V5 checkpoint and adapter package contracts

**Files:**
- Create: `saved_time_phase_operator_v4/instance_adaptation/__init__.py`
- Create: `saved_time_phase_operator_v4/instance_adaptation/contracts.py`
- Create: `tests/saved_time_phase_operator_v4/test_instance_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

```python
def test_onset_indices_are_exactly_first_two_saved_frames_after_t0():
    axis = torch.tensor([0.00, 0.01, 0.02, 0.03])
    assert onset_indices(axis, 0.011) == (2, 3)

def test_rejects_k1_out_of_range():
    with pytest.raises(ValueError, match="two onset frames"):
        onset_indices(torch.tensor([0.0]), 0.0)

def test_future_indices_are_strictly_after_k1():
    assert future_indices(6, (2, 3)).tolist() == [4, 5]

def test_snapshot_access_audit_rejects_later_true_frame():
    audit = SnapshotAccessAudit(allowed_indices=(2, 3))
    audit.read((2, 3))
    with pytest.raises(PermissionError, match="future truth"):
        audit.read((4,))
```

Run: `python3 -m pytest -q tests/saved_time_phase_operator_v4/test_instance_contracts.py`
Expected: FAIL because the contract types and functions do not exist.

- [ ] **Step 2: Implement the minimal immutable contracts**

Implement `onset_indices(time_s, t0_s) -> tuple[int, int]`, `future_indices(time_count, observed_indices) -> torch.Tensor`, and `SnapshotAccessAudit`. `onset_indices` must use `torch.searchsorted(time_s, torch.as_tensor(t0_s, dtype=time_s.dtype), right=False)`, return `(k0, k0+1)`, and reject non-monotonic axes, non-finite inputs, or an out-of-range `k1`. `SnapshotAccessAudit.read` must record every requested index and raise `PermissionError` for any index not equal to the two allowed indices. Add `SyntheticBridgeProvenance` with `synthetic=True`, source indices, and a SHA-256 digest of the generated tensor.

- [ ] **Step 3: Run the contract tests**

Run the pytest command above. Expected: all four tests PASS.

- [ ] **Step 4: Commit the contract boundary**

```bash
git add saved_time_phase_operator_v4/instance_adaptation tests/saved_time_phase_operator_v4/test_instance_contracts.py
git commit -m "feat: add V5 causal instance contracts"
```

## Task 2: Implement a two-frame-only deployment dataset

**Files:**
- Create: `saved_time_phase_operator_v4/instance_adaptation/data_guard.py`
- Create: `tests/saved_time_phase_operator_v4/test_instance_data_guard.py`

- [ ] **Step 1: Write tests for guarded reads**

```python
def test_guarded_record_exposes_metadata_and_two_frames_only(fake_h5):
    record = GuardedOnsetRecord(fake_h5, record_index=0, audit=SnapshotAccessAudit((2, 3)))
    assert record.observed_indices == (2, 3)
    assert record.observed_wavefield.shape[0] == 2
    assert set(record.public_keys) == {
        "velocity_mps", "source_parameters", "source_map", "time_s",
        "x_m", "z_m", "observed_wavefield", "observed_indices",
    }

def test_guarded_record_cannot_read_k1_plus_one(fake_h5):
    record = GuardedOnsetRecord(fake_h5, record_index=0, audit=SnapshotAccessAudit((2, 3)))
    with pytest.raises(PermissionError):
        record.read_truth_forbidden((4,))

def test_future_truth_replacement_does_not_change_adapter_inputs(fake_h5, mutated_h5):
    left = GuardedOnsetRecord(fake_h5, 0, SnapshotAccessAudit((2, 3)))
    right = GuardedOnsetRecord(mutated_h5, 0, SnapshotAccessAudit((2, 3)))
    assert torch.equal(left.observed_wavefield, right.observed_wavefield)
    assert left.input_digest == right.input_digest
```

- [ ] **Step 2: Implement the guard using the existing manifest**

Build `GuardedOnsetDataset(source_h5, manifest, split, sample_ids=None)` on top of `V3WavefieldDataset`, but never expose its `read_wavefield` method. For each record, read velocity/source metadata and exactly `wavefield[source_index, [k0,k1]]`; call `SnapshotAccessAudit.read` before the HDF5 slice. Store `input_digest` over velocity, source, time axis, and the two frames. Do not implement an arbitrary time-index or pressure accessor on the returned record.

- [ ] **Step 3: Run data-guard tests and HDF5 contract checks**

Run:

```bash
python3 -m pytest -q tests/saved_time_phase_operator_v4/test_instance_data_guard.py tests/test_lwc84_hdf5.py
```

Expected: all tests PASS; the mutated future slice produces an identical guarded input digest.

- [ ] **Step 4: Commit the guarded dataset**

```bash
git add saved_time_phase_operator_v4/instance_adaptation/data_guard.py tests/saved_time_phase_operator_v4/test_instance_data_guard.py
git commit -m "feat: guard V5 adaptation to onset snapshots"
```

## Task 3: Add zero-initialized onset conditioner and low-rank adapter

**Files:**
- Create: `saved_time_phase_operator_v4/instance_adaptation/adapters.py`
- Create: `tests/saved_time_phase_operator_v4/test_instance_adapters.py`

- [ ] **Step 1: Write identity, parameter-budget, and gradient tests**

```python
def test_zero_init_adapter_is_parent_identity(parent_and_record):
    parent, record = parent_and_record
    wrapper = OnsetAdaptedV5(parent, latent_dim=32, lora_rank=4)
    with torch.no_grad():
        base = parent.predict_wavefield(record.prepared, record.time_s)
        adapted = wrapper(record.velocity, record.source, record.time_s,
                          record.observed_wavefield, record.observed_indices)
    assert torch.allclose(base, adapted, atol=1e-7, rtol=1e-6)

def test_adapter_uses_less_than_one_percent_trainable_parameters(parent):
    wrapper = OnsetAdaptedV5(parent, latent_dim=32, lora_rank=4)
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapper.parameters())
    assert trainable / total < 0.01

def test_onset_loss_has_gradient_before_hard_projection(parent_and_record):
    parent, record = parent_and_record
    wrapper = OnsetAdaptedV5(parent, latent_dim=32, lora_rank=4)
    prediction = wrapper.raw_wavefield(record.velocity, record.source, record.time_s)
    loss = torch.nn.functional.mse_loss(
        prediction[:, list(record.observed_indices)], record.observed_wavefield
    )
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in wrapper.adapter_parameters())
```

- [ ] **Step 2: Implement the conditioner and low-rank residual**

Create `OnsetSnapshotConditioner` with a three-channel input `[velocity, observed_k0, observed_k1]`, source MLP features, adaptive pooling, and a latent output. Create `ZeroInitLoRA` for linear projections and `LowRankFieldResidual` that maps the latent plus velocity summary to a separable spatial basis and a 401-time coefficient. The correction output must be zero at construction. `OnsetAdaptedV5` freezes every parent parameter, computes the parent complete field, computes a correction from only velocity/source/two frames, and exposes `raw_wavefield(velocity, source, time_s)` plus `hard_project(prediction, observed_wavefield, observed_indices)`. The public forward applies hard projection only after the optimization loss has been computed by the trainer.

- [ ] **Step 3: Run adapter tests**

Run:

```bash
python3 -m pytest -q tests/saved_time_phase_operator_v4/test_instance_adapters.py
```

Expected: identity, budget, and gradient tests PASS on CPU.

- [ ] **Step 4: Commit the adapter**

```bash
git add saved_time_phase_operator_v4/instance_adaptation/adapters.py tests/saved_time_phase_operator_v4/test_instance_adapters.py
git commit -m "feat: add zero-init V5 onset adapter"
```

## Task 4: Add deterministic LWC-84 bridge and dimensionless losses

**Files:**
- Create: `saved_time_phase_operator_v4/instance_adaptation/bridge.py`
- Create: `saved_time_phase_operator_v4/instance_adaptation/losses.py`
- Create: `tests/saved_time_phase_operator_v4/test_instance_physics.py`

- [ ] **Step 1: Write bridge and loss tests**

```python
def test_bridge_is_marked_synthetic_and_never_reads_hdf5(fake_record):
    result = make_onset_bridge(fake_record.velocity_mps, fake_record.source_parameters,
                               fake_record.observed_wavefield, fake_record.observed_indices,
                               fake_record.time_s, steps=2, device="cpu")
    assert result.provenance.synthetic is True
    assert result.provenance.true_indices == (2, 3)

def test_zero_residual_for_matching_discrete_triplet():
    field = analytic_discrete_wavefield()
    assert lwc84_residual_loss(field, velocity=constant_velocity(), dt=0.0025,
                               dx=10.0, dz=10.0).item() < 1e-5

def test_loss_terms_are_finite_and_dimensionless(parent_and_record):
    terms = instance_loss_terms(
        raw_prediction=parent_and_record.prediction,
        target_observed=parent_and_record.observed_wavefield,
        bridge=parent_and_record.bridge,
        velocity=parent_and_record.velocity,
        source=parent_and_record.source,
        points=parent_and_record.points,
        weights=LossWeights(),
    )
    assert all(torch.isfinite(value) for value in terms.values())
```

- [ ] **Step 2: Implement bridge from existing LWC-84 primitives**

Reuse `src/fno_acoustic.data_generation.lwc84.lwc84_startup`, `lwc84_step`, CPML profile helpers,
and `BoundaryConfig`/`AcousticGrid` from the existing data-generation package. Use `k0/k1` to estimate
the startup state, generate at most four saved-time synthetic frames by default, and return
`BridgeResult(frames, time_indices, provenance, valid, failure_reason)`. If dimensions, CFL, or
boundary state are invalid, return `valid=False` and no bridge loss rather than fabricating labels.

- [ ] **Step 3: Implement losses with fixed collocation**

Implement `build_fixed_physics_points(time_count, observed_indices, count, seed)` and
`instance_loss_terms(raw_prediction, target_observed, bridge, velocity, source, points, weights)`.
The terms are observation Charbonnier, bridge relative L2, LWC-84 residual, spectral phase,
energy-envelope, and adapter anchor. Normalize each term by a detached reference norm. The function
must not accept a future truth tensor or a dataset object.

- [ ] **Step 4: Run physics tests**

Run:

```bash
python3 -m pytest -q tests/saved_time_phase_operator_v4/test_instance_physics.py tests/test_lwc84_core.py tests/test_lwc84_cpml.py
```

Expected: bridge provenance, finite-loss, and discrete residual tests PASS.

- [ ] **Step 5: Commit physics components**

```bash
git add saved_time_phase_operator_v4/instance_adaptation/bridge.py saved_time_phase_operator_v4/instance_adaptation/losses.py tests/saved_time_phase_operator_v4/test_instance_physics.py
git commit -m "feat: add deterministic onset physics losses"
```

## Task 5: Implement adaptation trainer, acceptance gate, and checkpoint format

**Files:**
- Create: `saved_time_phase_operator_v4/instance_adaptation/trainer.py`
- Create: `tests/saved_time_phase_operator_v4/test_instance_trainer.py`

- [ ] **Step 1: Write trainer and rollback tests**

```python
def test_trainer_updates_only_adapter_and_records_access(parent_and_record):
    result = adapt_instance(parent, record, adam_steps=2, lbfgs_steps=0)
    assert result.accessed_true_indices == (record.observed_indices[0], record.observed_indices[1])
    assert result.trainable_parameter_count < result.total_parameter_count / 100

def test_nonfinite_or_bad_physics_candidate_rolls_back(parent_and_record, monkeypatch):
    monkeypatch.setattr(
        "saved_time_phase_operator_v4.instance_adaptation.trainer.instance_loss_terms",
        lambda **kwargs: {"total": torch.tensor(float("nan"), requires_grad=True)},
    )
    result = adapt_instance(parent, record, adam_steps=1, lbfgs_steps=0)
    assert result.accepted is False
    assert result.rollback_reason == "nonfinite_loss"

def test_limited_lbfgs_uses_fixed_points(parent_and_record):
    result = adapt_instance(parent, record, adam_steps=1, lbfgs_steps=2)
    assert result.lbfgs_closure_calls >= 1
    assert result.future_truth_used is False
```

- [ ] **Step 2: Implement the two-stage optimizer**

Implement `adapt_instance(model, record, *, conditioner=None, adam_steps=12, lbfgs_steps=4,
                          learning_rate=2e-3, seed=17)` with 12 AdamW steps by default, fixed collocation points, gradient
clipping at 1.0, and a deterministic 3–5-iteration L-BFGS phase. Compute `raw_wavefield` for the
loss, then apply `hard_project` only for candidate output. Snapshot access must be performed through
`SnapshotAccessAudit`; no future HDF5 access is possible from the function signature. Return
`AdaptationResult` containing adapter state, 0-step and final physics diagnostics, timings, access log,
finite checks, and rollback reason.

- [ ] **Step 3: Implement the no-future acceptance gate**

Accept only if observed loss improves, physics residual improves by at least 5%, energy ratio lies in
`[0.5, 2.0]`, correction norm is bounded, all tensors are finite, and the audit contains only
`k0/k1`. Otherwise restore the cloned zero-step adapter state. Save `adapted.pt`, `adaptation.json`,
and a state hash.

- [ ] **Step 4: Run trainer tests**

Run:

```bash
python3 -m pytest -q tests/saved_time_phase_operator_v4/test_instance_trainer.py
```

Expected: update isolation, rollback, and fixed-L-BFGS tests PASS.

- [ ] **Step 5: Commit trainer and gate**

```bash
git add saved_time_phase_operator_v4/instance_adaptation/trainer.py tests/saved_time_phase_operator_v4/test_instance_trainer.py
git commit -m "feat: add guarded two-stage instance adaptation"
```

## Task 6: Add train-split onset-conditioner pretraining

**Files:**
- Create: `scripts/train_v5_onset_conditioner.py`
- Create: `configs/saved_time_v5/onset_conditioner.yaml`
- Create: `tests/saved_time_phase_operator_v4/test_conditioner_training.py`

- [ ] **Step 1: Write episode and split-isolation tests**

```python
def test_conditioner_episode_uses_only_train_group_ids(train_manifest):
    episodes = build_onset_episodes(train_manifest, split="train", seed=17)
    assert all(episode.split == "train" for episode in episodes)
    assert len({episode.group_id for episode in episodes}) == len(episodes)

def test_conditioner_checkpoint_contains_parent_identity(tmp_path):
    path = train_conditioner_smoke(
        config_path="configs/saved_time_v5/onset_conditioner.yaml",
        output=tmp_path / "conditioner.pt",
        device="cpu",
        max_steps=2,
    )
    payload = torch.load(path, weights_only=False)
    assert payload["parent_checkpoint_sha256"]
    assert payload["future_truth_used_only_for_train_episode"] is True
```

- [ ] **Step 2: Implement the offline episode trainer**

Load the selected parent checkpoint through the existing V5 builder, freeze the parent, and train the
conditioner on train records only. Each episode reads exactly the two onset frames for conditioning;
future train truth is used only as offline episode supervision and is never exposed by the deployment
dataset. Balance the three families, save the manifest/config/normalization hashes, and checkpoint the
best validation-free train objective without reading validation future truth.

- [ ] **Step 3: Run the conditioner smoke test**

Run:

```bash
python3 -m pytest -q tests/saved_time_phase_operator_v4/test_conditioner_training.py
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_v5_onset_conditioner.py --config configs/saved_time_v5/onset_conditioner.yaml --smoke --device cuda
```

Expected: CPU tests PASS and the CUDA smoke writes a finite checkpoint with the parent identity.

- [ ] **Step 4: Commit the conditioner trainer**

```bash
git add scripts/train_v5_onset_conditioner.py configs/saved_time_v5/onset_conditioner.yaml tests/saved_time_phase_operator_v4/test_conditioner_training.py
git commit -m "feat: pretrain V5 onset conditioner"
```

## Task 7: Add sealed evaluator, exact-time metrics, and publication figures

**Files:**
- Create: `scripts/evaluate_v5_instance_adaptation.py`
- Create: `saved_time_phase_operator_v4/instance_adaptation/visualization.py`
- Create: `tests/saved_time_phase_operator_v4/test_instance_evaluation.py`

- [ ] **Step 1: Write evaluator sealing tests**

```python
def test_evaluator_refuses_unsealed_future_truth(adapted_artifact, truth_h5):
    with pytest.raises(RuntimeError, match="sealed"):
        evaluate_after_adaptation(adapted_artifact, truth_h5, sealed=False)

def test_evaluator_streams_all_401_saved_times(adapted_artifact, truth_h5):
    report = evaluate_after_adaptation(adapted_artifact, truth_h5, sealed=True)
    assert report["metrics"]["unique_time_index_count"] == 401

def test_receiver_traces_are_derived_from_full_field(adapted_artifact):
    result = derive_receivers_from_field(full_field, receiver_indices)
    assert result.shape == (len(receiver_indices), 401)
```

- [ ] **Step 2: Implement sealed all-time evaluation**

Load the parent/adapted checkpoint and predict the full saved axis in blocks. Only after adapter state,
access audit, and acceptance JSON are hash-verified may the evaluator open later truth. Use
`ExactWavefieldMetricAccumulator` with `k > k1` masks, family grouping, phase/spectrum/time-bin
metrics, and per-instance wall-clock measurements. Write one JSON report per instance and one aggregate
report over the nine sealed instances.

- [ ] **Step 3: Implement field and receiver plots**

Choose the three post-onset frame indices deterministically from each instance's saved axis before
looking at predictions. Render true/parent/adapted/error columns with shared symmetric target scales,
and derive nine fixed near-surface receiver traces from the predicted/true full fields. Use the
Okabe–Ito palette, Matplotlib, and save each figure as both PDF and 300 dpi PNG. Keep plot data and
receiver indices in JSON.

- [ ] **Step 4: Run evaluator tests**

Run:

```bash
python3 -m pytest -q tests/saved_time_phase_operator_v4/test_instance_evaluation.py tests/saved_time_phase_operator_v4/test_streaming_metrics.py
```

Expected: sealing, 401-time coverage, and receiver-derivation tests PASS.

- [ ] **Step 5: Commit evaluation and figures**

```bash
git add scripts/evaluate_v5_instance_adaptation.py saved_time_phase_operator_v4/instance_adaptation/visualization.py tests/saved_time_phase_operator_v4/test_instance_evaluation.py
git commit -m "feat: add sealed V5 adaptation evaluation figures"
```

## Task 8: Add three-family manifest and command-line orchestration

**Files:**
- Create: `configs/saved_time_v5/instance_adaptation.yaml`
- Create: `scripts/run_v5_instance_adaptation.py`
- Create: `tests/saved_time_phase_operator_v4/test_instance_cli.py`

- [ ] **Step 1: Write CLI dry-run tests**

```python
def test_cli_manifest_has_three_families_and_nine_instances(tmp_path):
    manifest = build_instance_manifest(validation_cache, seed=17)
    assert {row.medium_type for row in manifest} == {"uniform", "layered", "marmousi"}
    assert {row.medium_type: sum(x.medium_type == row.medium_type for x in manifest)
            for row in manifest} == {"uniform": 3, "layered": 3, "marmousi": 3}

def test_cli_dry_run_does_not_open_future_pressure(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/run_v5_instance_adaptation.py",
         "--config", "configs/saved_time_v5/instance_adaptation.yaml",
         "--dry-run", "--output-dir", str(tmp_path / "dry_run")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "future_truth_opened=false" in result.stdout
```

- [ ] **Step 2: Implement manifest selection and orchestration**

Select exactly three validation records per family, stratified by source position/frequency/onset,
write the frozen manifest before adaptation, and run methods A/B/C with fixed seeds. The CLI must
support `--smoke`, `--sample-id`, `--method`, `--dry-run`, `--device`, and `--output-dir`. It must
refuse a manifest with anomaly records, duplicate group IDs, or a non-V5 parent identity.

- [ ] **Step 3: Run dry-run and CPU CLI tests**

Run:

```bash
python3 -m pytest -q tests/saved_time_phase_operator_v4/test_instance_cli.py
python3 scripts/run_v5_instance_adaptation.py --config configs/saved_time_v5/instance_adaptation.yaml --dry-run
```

Expected: exactly `3/3/3` family counts, two allowed truth indices per record, and no future truth opened.

- [ ] **Step 4: Commit orchestration**

```bash
git add configs/saved_time_v5/instance_adaptation.yaml scripts/run_v5_instance_adaptation.py tests/saved_time_phase_operator_v4/test_instance_cli.py
git commit -m "feat: orchestrate three-family V5 adaptation"
```

## Task 9: Execute staged validation and generate the requested artifacts

**Files:**
- Create: `artifacts/v5_instance_adaptation/pilot/` and `artifacts/v5_instance_adaptation/formal/`
- Create: `docs/superpowers/reports/2026-07-18-v5-instance-adaptation-results.md`

- [ ] **Step 1: Run the existing full test subset**

```bash
python3 -m pytest -q tests/saved_time_phase_operator_v4 tests/grouped_ufno_mionet_v3/test_records.py tests/test_lwc84_core.py tests/test_lwc84_cpml.py
```

Expected: all existing and new tests PASS before GPU experiments.

- [ ] **Step 2: Select and seal the parent checkpoint**

Run the existing all-record evaluator at epoch 10. If epoch 10 is not yet available, wait for the
training process to write it; do not use the panel report. Select the lowest aggregate full-validation
checkpoint and write `parent_identity.json` with checkpoint SHA-256, manifest, time-axis, normalizer,
and Git hashes.

- [ ] **Step 3: Run one-instance-per-family CUDA pilot**

```bash
mkdir -p artifacts/v5_instance_adaptation/parent
cp /data/jiayh/saved_time_v5_training_recovery_batch48/run/best.pt \
  artifacts/v5_instance_adaptation/parent/best.pt
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_v5_instance_adaptation.py \
  --config configs/saved_time_v5/instance_adaptation.yaml \
  --parent-checkpoint artifacts/v5_instance_adaptation/parent/best.pt --pilot --device cuda \
  --output-dir artifacts/v5_instance_adaptation/pilot
```

Expected: one Uniform, one Layered, one Marmousi instance; all outputs finite; only two true indices
accessed; adapter acceptance or rollback recorded; full 401-time predictions written.

- [ ] **Step 4: Run the nine-instance experiment**

```bash
CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/run_v5_instance_adaptation.py \
  --config configs/saved_time_v5/instance_adaptation.yaml \
  --parent-checkpoint artifacts/v5_instance_adaptation/parent/best.pt --device cuda \
  --output-dir artifacts/v5_instance_adaptation/formal \
  > artifacts/v5_instance_adaptation/formal.log 2>&1 &
```

Monitor only at epoch/instance boundaries, record GPU utilization and stage timings, and preserve the
log if one candidate rolls back. Do not terminate the unrelated long pretraining process.

- [ ] **Step 5: Generate and inspect figures**

Confirm each family has PDF/PNG wavefield snapshots and receiver waveform plots, each report includes
future full-field metrics, and all 401 stored time indices are present. Inspect color scales, source
marker, receiver locations, legends, and residual panels before writing conclusions.

- [ ] **Step 6: Write the results report and commit only source/report changes**

The report must separate parent baseline, latent-only, recommended adapter, and full-decoder control;
show per-family and per-instance future errors, acceptance/rollback counts, runtime, speedup, and the
exact figure paths. If the 10% target is missed, state the failed family and the observed Pareto point.

```bash
git add docs/superpowers/reports/2026-07-18-v5-instance-adaptation-results.md
git commit -m "report: evaluate V5 causal instance adaptation"
```

## Self-review checklist

- Spec coverage: Tasks 1–2 enforce exactly two early true frames and no later access; Task 3 implements
  zero-init low-rank adaptation; Task 4 supplies synthetic-only bridge and physics losses; Task 5 adds
  AdamW/L-BFGS, acceptance, rollback, and audit; Task 6 covers the train-only conditioner; Tasks 7–8
  cover sealed all-401 evaluation, receivers, figures, and 3/3/3 family selection; Task 9 runs the
  staged experiment and report.
- Placeholder scan: no task uses TODO/TBD or an unspecified validation command; every code-changing task
  names exact files, test commands, expected outcomes, and a commit.
- Type consistency: `onset_indices` returns `(k0,k1)` throughout; `GuardedOnsetRecord.observed_indices`,
  `AdaptationResult.accessed_true_indices`, and evaluator masks all use the same tuple of integer saved
  indices; synthetic bridge frames carry `SyntheticBridgeProvenance` and never enter the truth audit.
- Scope check: the plan stays within the approved V5 adaptation, three allowed families, full saved-time
  wavefield output, and derived receiver diagnostics. FWI, anomaly media, interpolation, and GitHub push
  remain excluded.
