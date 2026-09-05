# CLFC Efficient Two-Frame Instance Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, closed-form 12-scalar CLFC adapter that reads exactly the first two saved frames at or after the registered source onset, preserves the frozen epoch-40 parent, seals its state before future truth is opened, and evaluates one Uniform, Layered, and Marmousi instance.

**Architecture:** Extend the existing guarded-record boundary with an explicit zero-lead onset mode, capture the frozen parent field and residual local-field activation in one forward pass, then fit bounded ridge coefficients from only the two guarded frames. Store and hash the calibration decision before a separate sealed evaluator reads later truth; rejection always returns the exact parent. The downloaded source tree has no `.git`, so the plan uses focused test checkpoints and SHA-256 provenance instead of inventing commits.

**Tech Stack:** Python 3.10, PyTorch 2.7, NumPy, h5py, PyYAML, pytest, Matplotlib, existing saved-time V4/V5 model and evaluation utilities.

---

## File map

- Modify `saved_time_phase_operator_v4/instance_adaptation/data_guard.py`: make the onset lead explicit while preserving the current default for older V5 callers.
- Create `saved_time_phase_operator_v4/instance_adaptation/local_field_cache.py`: capture and validate detached parent/local-field decomposition, causal gate, and free-surface factor.
- Create `saved_time_phase_operator_v4/instance_adaptation/clfc.py`: construct three spatial bands and 12 bases, perform float64 ridge cross-fitting, apply trust gates, and return an immutable result.
- Create `saved_time_phase_operator_v4/instance_adaptation/sealed_evaluation.py`: atomically seal, hash, reload, and verify a calibration artifact before future truth access.
- Create `scripts/run_clfc_two_frame_evaluation.py`: strict parent loading, deterministic selection, guarded adaptation, state sealing, post-seal evaluation, metrics, fields, and figures.
- Create `configs/saved_time_v5/clfc_epoch40_smoke3.yaml`: bind the local VDS, travel-time file, epoch-40 checkpoint identities, solver grid, gates, and seed 372.
- Create `tests/saved_time_phase_operator_v4/test_clfc_data_guard.py`: strict `time >= t0` boundary tests.
- Create `tests/saved_time_phase_operator_v4/test_local_field_cache.py`: hook, detach, shape, causality, free-surface, and one-forward tests.
- Create `tests/saved_time_phase_operator_v4/test_clfc.py`: bands, basis order, closed-form recovery, cross-fit, bounds, rejection, and determinism tests.
- Create `tests/saved_time_phase_operator_v4/test_clfc_sealed_evaluation.py`: state-hash tamper and future-mutation invariance tests.
- Create `tests/saved_time_phase_operator_v4/test_clfc_cli.py`: dry-run selection and pre-evaluation seal-order contract.

### Task 1: Enforce the literal two-post-onset boundary

**Files:**
- Modify: `saved_time_phase_operator_v4/instance_adaptation/data_guard.py`
- Create: `tests/saved_time_phase_operator_v4/test_clfc_data_guard.py`

- [x] **Step 1: Write the failing strict-onset unit tests**

```python
import torch

from saved_time_phase_operator_v4.instance_adaptation.contracts import onset_indices


def test_zero_lead_selects_first_two_frames_at_or_after_t0():
    time_s = torch.arange(0.0, 0.021, 0.005)
    assert onset_indices(time_s, t0_s=0.011, f0_hz=10.0, lead_cycles=0.0) == (3, 4)


def test_one_cycle_default_remains_backward_compatible():
    time_s = torch.arange(0.0, 0.201, 0.005)
    assert onset_indices(time_s, t0_s=0.111, f0_hz=10.0) == (3, 4)
```

Add a monkeypatched dataset test that instantiates
`GuardedOnsetDataset(source_h5, manifest, split="validation",
onset_lead_cycles=0.0)` and asserts the returned record uses the two strict
indices and its audit requests only those indices.

- [x] **Step 2: Run the tests and verify the dataset constructor fails**

Run:

```bash
pytest -q tests/saved_time_phase_operator_v4/test_clfc_data_guard.py
```

Expected: FAIL because `GuardedOnsetDataset.__init__` does not accept
`onset_lead_cycles`.

- [x] **Step 3: Add the explicit dataset option**

Store a finite nonnegative `self.onset_lead_cycles` in the constructor and call:

```python
observed = onset_indices(
    metadata_record.time_s,
    t0_s=float(metadata_record.source_parameters[3]),
    f0_hz=float(metadata_record.source_parameters[2]),
    lead_cycles=self.onset_lead_cycles,
)
```

Keep the constructor default at `1.0` so existing V5 behavior is unchanged.
CLFC must pass `0.0` explicitly.

- [x] **Step 4: Run strict and legacy guard tests**

Run:

```bash
pytest -q \
  tests/saved_time_phase_operator_v4/test_clfc_data_guard.py \
  tests/saved_time_phase_operator_v4/test_instance_data_guard.py
```

Expected: all tests PASS; the strict test records only adjacent indices with
`time_s[k0] >= t0`.

### Task 2: Capture one detached parent/local-field decomposition

**Files:**
- Create: `saved_time_phase_operator_v4/instance_adaptation/local_field_cache.py`
- Create: `tests/saved_time_phase_operator_v4/test_local_field_cache.py`

- [x] **Step 1: Write cache invariant tests**

Use a tiny fake parent whose `local_field` is a real `nn.Module` and whose
`predict_wavefield` calls it in two time blocks. Assert:

```python
cache = capture_parent_local_field(
    model, normalizer=None, record=record, device=torch.device("cpu"), time_block=2
)
assert cache.parent_field.shape == cache.local_field.shape == (1, 6, 4, 4)
assert cache.causal_gate.shape == (1, 6, 4, 4)
assert cache.free_surface_factor.shape == (1, 1, 4, 1)
assert cache.parent_field.dtype == cache.local_field.dtype == torch.float32
assert not cache.parent_field.requires_grad
assert not cache.local_field.requires_grad
assert torch.count_nonzero(cache.local_field[:, :, 0, :]) == 0
assert model.predict_calls == 1
```

Also test that a missing local-field module, a nonfinite activation, an
activation count inconsistent with the time axis, or a nonzero free-surface
top row raises `ValueError`.

- [x] **Step 2: Run the cache tests and verify import failure**

Run:

```bash
pytest -q tests/saved_time_phase_operator_v4/test_local_field_cache.py
```

Expected: FAIL with `ModuleNotFoundError` for `local_field_cache`.

- [x] **Step 3: Implement the cache interface**

Define:

```python
@dataclass(frozen=True)
class ParentLocalFieldCache:
    parent_field: torch.Tensor
    local_field: torch.Tensor
    causal_gate: torch.Tensor
    free_surface_factor: torch.Tensor
    time_s: torch.Tensor
    parent_forward_seconds: float
    cache_seconds: float


def capture_parent_local_field(
    model,
    normalizer,
    record: GuardedOnsetRecord,
    device: torch.device,
    *,
    time_block: int = 1,
) -> ParentLocalFieldCache
```

Register a temporary forward hook on `model.local_field`, run the existing
source preparation and `predict_wavefield` path once, remove the hook in
`finally`, concatenate time blocks, detach to contiguous float32, and multiply
the captured residual by the same
`grouped_ufno_mionet_v3.model.operator.free_surface_factor` used by the parent.
Reconstruct the exact gate as
`sigmoid((time - t0 - travel_time) / model.local_field.causal_width_s)`.
Validate finite tensors, registered time-axis equality, zero top row, and shape
identity.

- [x] **Step 4: Run cache tests**

Run:

```bash
pytest -q tests/saved_time_phase_operator_v4/test_local_field_cache.py
```

Expected: all tests PASS and the fake parent's prediction counter equals one.

### Task 3: Implement the deterministic CLFC numerical core

**Files:**
- Create: `saved_time_phase_operator_v4/instance_adaptation/clfc.py`
- Create: `tests/saved_time_phase_operator_v4/test_clfc.py`

- [x] **Step 1: Write band and basis tests**

Test that `radial_cosine_band_masks(32, 24)` returns three finite masks whose
sum is one within `1e-6`, low contains DC, high contains the highest radial
frequencies, and repeated calls are identical. For a synthetic traveling field,
assert `build_clfc_bases` returns shape `[12, time, z, x]` in this fixed order:

```text
low amplitude constant, low amplitude linear,
low derivative constant, low derivative linear,
middle amplitude constant, middle amplitude linear,
middle derivative constant, middle derivative linear,
high amplitude constant, high amplitude linear,
high derivative constant, high derivative linear
```

Assert every basis has zero top row and is zero wherever the causal gate is
below the configured support tolerance.

- [x] **Step 2: Write closed-form recovery and cross-fit tests**

Create two 8-by-8 observed frames from a known 12-vector with independent basis
columns. Assert normalized float64 ridge recovers the identifiable coefficients
within tolerance, chooses a lambda from
`(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10)`, reports both held-out losses, and
is bitwise deterministic on CPU.

- [x] **Step 3: Write trust-region and rollback tests**

Use parametrized cases for:

- condition number above `1e6`;
- nonfinite coefficient;
- effective amplitude above `0.15`;
- phase shift above `0.5 * dt`;
- correction norm ratio above `0.25`;
- energy ratio outside `[0.75, 1.25]`;
- causal-support violation;
- nonzero top row;
- audit indices beyond `(k0, k1)`.

Each case must return `accepted=False`, `shrinkage=0.0`, and an adapted field
exactly equal to the parent. A passing synthetic case must choose the largest
allowed shrinkage from `(1.0, 0.5, 0.25, 0.125, 0.0)`.

- [x] **Step 4: Run the numerical tests and verify import failure**

Run:

```bash
pytest -q tests/saved_time_phase_operator_v4/test_clfc.py
```

Expected: FAIL with `ModuleNotFoundError` for `clfc`.

- [x] **Step 5: Implement immutable configuration and result types**

Define:

```python
@dataclass(frozen=True)
class CLFCConfig:
    lambdas: Sequence[float] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    shrinkages: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0)
    maximum_condition_number: float = 1e6
    maximum_amplitude_fraction: float = 0.15
    maximum_phase_fraction_of_dt: float = 0.5
    maximum_correction_norm_ratio: float = 0.25
    minimum_energy_ratio: float = 0.75
    maximum_energy_ratio: float = 1.25
    causal_support_tolerance: float = 1e-6


@dataclass(frozen=True)
class CLFCResult:
    coefficients: Sequence[float]
    column_scales: Sequence[float]
    ridge_lambda: float
    shrinkage: float
    condition_number: float
    crossfit_losses: tuple[float, float]
    gate_diagnostics: dict[str, object]
    observed_indices: tuple[int, int]
    accessed_indices: Sequence[int]
    accepted: bool
    rollback_reason: str | None
    fit_seconds: float
    state_digest: str

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)
```

- [x] **Step 6: Implement bands, bases, ridge selection, and gates**

Implement these pure functions:

```python
radial_cosine_band_masks(height, width, *, device, dtype) -> torch.Tensor
split_spatial_bands(local_field) -> torch.Tensor
centered_time_derivative(field, time_s) -> torch.Tensor
build_clfc_bases(cache, observed_indices) -> torch.Tensor
solve_ridge(design, target, ridge_lambda) -> tuple[torch.Tensor, torch.Tensor, float]
select_crossfit_lambda(design_by_frame, target_by_frame, lambdas) -> Sequence[float]
fit_clfc(cache, record, config=CLFCConfig()) -> tuple[CLFCResult, torch.Tensor]
```

Normalize design columns by detached RMS, solve in float64 with
`torch.linalg.solve`, break lambda ties by the larger lambda, refit both frames,
and evaluate shrinkages from largest to smallest. Fit target is only
`observed_truth - parent[k0:k1+1]`. The time-linear factor is exactly
`clip((t-time[k1])/(time[-1]-time[k1]), 0, 1)`. Hard-project the two observed
frames only after raw-fit diagnostics are stored. When rejected, return an
exact clone of the parent without hard projection.

- [x] **Step 7: Run numerical tests**

Run:

```bash
pytest -q tests/saved_time_phase_operator_v4/test_clfc.py
```

Expected: all tests PASS with no optimizer or autograd graph involved.

### Task 4: Seal state before any future-truth read

**Files:**
- Create: `saved_time_phase_operator_v4/instance_adaptation/sealed_evaluation.py`
- Create: `tests/saved_time_phase_operator_v4/test_clfc_sealed_evaluation.py`

- [x] **Step 1: Write seal/tamper tests**

Construct a minimal calibration payload and assert:

```python
sealed = seal_calibration_state(payload, output_path)
assert output_path.exists()
assert verify_calibration_state(output_path).state_digest == sealed.state_digest
```

Flip one coefficient in the saved payload and assert verification raises
`ValueError("sealed calibration digest mismatch")`. Assert a payload whose
access audit includes `k1 + 1` is rejected before writing.

- [x] **Step 2: Write the future-mutation invariance test**

Build two synthetic HDF5 files with identical inputs and frames `k0/k1`, replace
all frames after `k1` in the second file with independent random values, run
only guarded CLFC adaptation, and assert equality of:

```python
assert left.coefficients == right.coefficients
assert left.accepted == right.accepted
assert left.accessed_indices == right.accessed_indices == (k0, k1)
assert left.state_digest == right.state_digest
```

Do not invoke the evaluator in this test.

- [x] **Step 3: Run tests and verify import failure**

Run:

```bash
pytest -q tests/saved_time_phase_operator_v4/test_clfc_sealed_evaluation.py
```

Expected: FAIL with `ModuleNotFoundError` for `sealed_evaluation`.

- [x] **Step 4: Implement canonical hashing and atomic writes**

Define a canonical JSON serializer that sorts dictionary keys, rejects NaN and
Infinity, and hashes UTF-8 bytes with SHA-256. Write to a sibling temporary file,
`flush` and `fsync`, then `os.replace` it into place. Define:

```python
@dataclass(frozen=True)
class SealedCalibrationState:
    payload: dict[str, object]
    state_digest: str
    path: Path


seal_calibration_state(payload, path) -> SealedCalibrationState
verify_calibration_state(path) -> SealedCalibrationState
```

Require `access_audit.future_truth_used is False` and requested indices equal
the observed pair. The evaluator must call `verify_calibration_state` before
opening the HDF5 wavefield dataset.

- [x] **Step 5: Run seal and mutation tests**

Run:

```bash
pytest -q tests/saved_time_phase_operator_v4/test_clfc_sealed_evaluation.py
```

Expected: all tests PASS; mutation of every future frame leaves adaptation state
unchanged.

### Task 5: Add the strict three-family CLFC runner

**Files:**
- Create: `scripts/run_clfc_two_frame_evaluation.py`
- Create: `configs/saved_time_v5/clfc_epoch40_smoke3.yaml`
- Create: `tests/saved_time_phase_operator_v4/test_clfc_cli.py`

- [x] **Step 1: Write CLI contract tests**

Call `main` with `--dry-run` and assert the fixed seed selects exactly:

```text
validation_uniform_00003
validation_layered_00082
validation_marmousi_00087
```

Assert the dry-run payload contains `onset_lead_cycles: 0.0`,
`allowed_true_snapshot_count: 2`, and `future_truth_opened: false`. Monkeypatch
`seal_calibration_state` and `_read_future_truth` to record events, run one tiny
instance, and assert the event order is `["seal", "future_truth_read"]`.

- [x] **Step 2: Run the CLI tests and verify script import failure**

Run:

```bash
pytest -q tests/saved_time_phase_operator_v4/test_clfc_cli.py
```

Expected: FAIL because `run_clfc_two_frame_evaluation.py` does not exist.

- [x] **Step 3: Write the bound configuration**

The YAML must bind:

```yaml
source_h5: /home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5
travel_time_h5: /home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5
parent_operator_config: /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/artifacts/parent_epoch40_smoke3/local_parent_loader.yaml
parent_checkpoint: /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/pretraining/gate4_long/run/best.pt
parent_checkpoint_identity: /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/pretraining/gate4_long/run/run_identity.json
expected_parent_checkpoint_sha256: 0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f
seed: 372
per_family: 1
onset_lead_cycles: 0.0
time_block: 1
```

Include the exact lambda, shrinkage, amplitude, phase, condition, correction
norm, and energy gates from the specification.

- [x] **Step 4: Implement runner phases**

Implement explicit phases:

1. load and validate config, source/travel/checkpoint identities;
2. select and atomically write `instance_manifest.json`;
3. create `GuardedOnsetDataset(source_h5, manifest, split="validation",
   sample_ids=sample_ids, travel_time_h5=travel_time_h5,
   onset_lead_cycles=0.0)`;
4. run `capture_parent_local_field` once per record;
5. run `fit_clfc` using only the guarded record;
6. atomically write and verify `calibration.json`;
7. save `fields.pt` containing parent, accepted-or-rollback field, and state hash;
8. only then call `_read_future_truth`;
9. compute paired future field, time-bin, spectrum, phase, per-time P50/P95,
   nine-receiver relative-L2/NRMSE, runtime, RAM, and CUDA peak-memory metrics;
10. reuse existing plotting utilities to write PNG and PDF comparisons;
11. atomically write per-record `evaluation.json` and aggregate `summary.json`.

Reject attempts to write under the pretraining artifact path. Record both the
remote checkpoint manifest digest and audited local mount-relocation digest.

- [x] **Step 5: Run CLI tests**

Run:

```bash
pytest -q tests/saved_time_phase_operator_v4/test_clfc_cli.py
```

Expected: all tests PASS and the dry run reports three fixed sample IDs without
opening future truth.

### Task 6: Focused verification before GPU evaluation

**Files:**
- Verify all files created or modified above.

- [x] **Step 1: Run the focused CLFC suite**

Run:

```bash
pytest -q \
  tests/saved_time_phase_operator_v4/test_clfc_data_guard.py \
  tests/saved_time_phase_operator_v4/test_local_field_cache.py \
  tests/saved_time_phase_operator_v4/test_clfc.py \
  tests/saved_time_phase_operator_v4/test_clfc_sealed_evaluation.py \
  tests/saved_time_phase_operator_v4/test_clfc_cli.py
```

Expected: all tests PASS.

- [x] **Step 2: Run relevant legacy regression tests**

Run:

```bash
pytest -q \
  tests/saved_time_phase_operator_v4/test_instance_data_guard.py \
  tests/saved_time_phase_operator_v4/test_instance_adapters.py \
  tests/saved_time_phase_operator_v4/test_full_support_runner.py
```

Expected: all tests PASS; the default one-cycle V5 boundary and parent loading
remain backward compatible.

- [x] **Step 3: Verify strict parent load and immutable checkpoint**

Load the epoch-40 checkpoint on CPU through the same runner code and assert the
model state loads strictly. Record SHA-256 before GPU evaluation:

```bash
sha256sum /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/pretraining/gate4_long/run/best.pt
```

Expected:

```text
0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f
```

### Task 7: Run strict smoke3 and decide the expansion gate

**Files:**
- Create outputs under `artifacts/clfc_two_frame_epoch40/smoke3`
- Modify: `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/SYNC_MANIFEST.md`

- [x] **Step 1: Run a sealed dry run**

Run:

```bash
python scripts/run_clfc_two_frame_evaluation.py \
  --config configs/saved_time_v5/clfc_epoch40_smoke3.yaml \
  --output-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/artifacts/clfc_two_frame_epoch40/smoke3 \
  --device cuda \
  --dry-run
```

Expected: the three fixed IDs, strict onset indices computed with zero lead,
two allowed frames, and `future_truth_opened=false`.

- [x] **Step 2: Run the three-record GPU evaluation**

Run the same command without `--dry-run`. Expected files for every sample:

```text
calibration.json
evaluation.json
fields.pt
wavefield_comparison.png
wavefield_comparison.pdf
receiver_waveforms.png
receiver_waveforms.pdf
```

Expected top-level files:

```text
instance_manifest.json
summary.json
```

- [x] **Step 3: Verify artifacts and research gate**

Assert:

- every observed pair satisfies `time[k0] >= t0` and `time[k0-1] < t0`;
- every requested truth index is exactly `k0` or `k1`;
- all state hashes verify and all reported numbers are finite;
- no sample regresses future full-field relative L2 by more than 1%;
- aggregate future full-field relative L2 improves by at least 3%;
- median CLFC overhead excluding parent forward is below 10 seconds;
- all PNG/PDF files are nonempty and one wavefield/receiver pair is visually
  inspected.

Set `expand_to_nine=true` only if every gate passes. Do not launch the nine-case
run in this task because the user requested three media examples first.

- [x] **Step 4: Recheck parent immutability and update provenance**

Recompute the checkpoint SHA-256 and require the same
`0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f`
value. Update `SYNC_MANIFEST.md` with strict-onset semantics,
source/config hashes, test command/results, output directory, three sample IDs,
and the smoke expansion decision.

## Self-review

- Spec coverage: Tasks 1–7 cover literal onset selection, one-pass cache,
  three bands and twelve coefficients, float64 cross-fitting, all eight gates,
  rollback, atomic sealing, future mutation, strict identity loading, metrics,
  plots, timings, and the three-to-nine research gate.
- Placeholder scan: every code-changing step names the exact interface,
  validation rule, command, and expected result.
- Type consistency: `ParentLocalFieldCache`, `CLFCConfig`, `CLFCResult`, and
  `SealedCalibrationState` are introduced once and used under the same names and
  shapes throughout.
- Repository constraint: no commit step is included because `.git` is absent;
  focused test results, atomic artifacts, and SHA-256 hashes replace commit
  checkpoints without creating false history.
