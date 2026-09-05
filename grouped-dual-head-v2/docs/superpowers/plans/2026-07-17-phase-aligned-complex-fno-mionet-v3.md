# Phase-Aligned Complex-FNO Multi-Input DeepONet V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify a three-medium-family, single-source acoustic neural operator that predicts arbitrary point values and complete `201×201` wavefield snapshots at arbitrary requested times, then pass the one-record and balanced nine-record accuracy gates before any pilot is allowed.

**Architecture:** A true learned-complex multiscale FNO encodes velocity once; independent velocity, source, differentiable ray-time, and periodic coordinate branches are fused by a multi-input DeepONet/MIONet. V2's grouped medium reuse, multiscale local pyramid, source-map/local-medium branch, direct local query residual, source/time FiLM dense conditioning, exact free-surface factor, and dual true-target supervision are retained. The shared continuous operator serves point queries, while a time-conditioned learned-complex spectral renderer produces coherent full-grid snapshots. No receiver observation enters the model or primary loss.

**Tech Stack:** Python 3.12, PyTorch, h5py, NumPy, PyYAML, pytest, Matplotlib, CUDA/BF16 for eligible local layers, FP32 FFT/loss/decoding.

**Approved design:** `docs/superpowers/specs/2026-07-17-phase-aligned-complex-fno-mionet-v3-design.md`

---

## Task 1: Establish the V3 package, configuration, and three-family manifest

**Files:**

- Create: `grouped_ufno_mionet_v3/__init__.py`
- Create: `grouped_ufno_mionet_v3/config.py`
- Create: `grouped_ufno_mionet_v3/data/__init__.py`
- Create: `grouped_ufno_mionet_v3/data/index.py`
- Create: `scripts/build_grouped_v3_manifest.py`
- Create: `configs/grouped_v3/smoke.yaml`
- Test: `tests/grouped_ufno_mionet_v3/test_config_and_index.py`

- [ ] Write the failing tests for the immutable family allowlist, exact filtered counts, stable manifest digest, and hard rejection of `anomaly`.

```python
def test_v3_index_excludes_anomaly_and_matches_census(v3_source_h5):
    manifest = build_manifest(v3_source_h5)
    assert manifest.allowed_medium_types == ("uniform", "layered", "marmousi")
    assert manifest.counts_after["train"] == 2240
    assert manifest.counts_after["validation"] == 480
    assert "anomaly" not in manifest.indices_by_family

def test_assert_allowed_families_is_a_hard_error():
    with pytest.raises(ValueError, match="anomaly"):
        assert_allowed_families(["uniform", "anomaly"])
```

- [ ] Run `pytest -q tests/grouped_ufno_mionet_v3/test_config_and_index.py` and confirm collection/import fails because V3 does not exist.
- [ ] Implement frozen dataclass configs, decode HDF5 byte/string fields, split filtering, pre/post census, source identity metadata, canonical JSON serialization, and SHA-256 manifest digest. Do not mutate the source HDF5 or V2 cache.
- [ ] Implement an atomic CLI writer using temporary-file plus `os.replace`; write sample indices and record IDs without loading wavefields.
- [ ] Run the focused test and `python scripts/build_grouped_v3_manifest.py --config configs/grouped_v3/smoke.yaml --dry-run`; expected output contains train `2240`, validation `480`, anomaly `0` after filtering.
- [ ] Commit: `feat(v3): add three-family data manifest contract`

## Task 2: Add filtered record loading and continuous-time interpolation

**Files:**

- Create: `grouped_ufno_mionet_v3/data/records.py`
- Create: `grouped_ufno_mionet_v3/data/batch.py`
- Create: `grouped_ufno_mionet_v3/normalization.py`
- Test: `tests/grouped_ufno_mionet_v3/test_records.py`
- Test: `tests/grouped_ufno_mionet_v3/test_normalization.py`

- [ ] Write failing synthetic-HDF5 tests proving exact frame reads, adjacent-frame interpolation, one-source-per-record preservation, source amplitude application exactly once, and no validation values in fitted normalization statistics.

```python
target = dataset.read_wavefield(record=0, times=torch.tensor([0.25]))
assert target.left_index.item() == 0
assert target.right_index.item() == 1
assert target.alpha.item() == pytest.approx(0.25)
torch.testing.assert_close(target.values, 0.75 * frame0 + 0.25 * frame1)
```

- [ ] Run the two focused files and confirm missing-module failures.
- [ ] Implement process-local lazy HDF5 handles, sorted unique frame reads, exact/interpolated target metadata, deterministic early/middle/late sampling, and `record_to_medium` collation. Return raw velocity and physical source parameters alongside normalized targets.
- [ ] Implement train-only normalization manifest binding. Reuse a prior numerical pressure scale only when an explicit streaming audit falls inside configured tolerance; otherwise fit and record a new scale.
- [ ] Verify with `pytest -q tests/grouped_ufno_mionet_v3/test_records.py tests/grouped_ufno_mionet_v3/test_normalization.py`.
- [ ] Commit: `feat(v3): load filtered exact and continuous wavefield targets`

## Task 3: Implement a true learned-complex spectral contraction

**Files:**

- Create: `grouped_ufno_mionet_v3/model/__init__.py`
- Create: `grouped_ufno_mionet_v3/model/spectral.py`
- Test: `tests/grouped_ufno_mionet_v3/test_spectral.py`

- [ ] Write failing tests comparing the layer to explicit `rfft2`/complex `einsum`/`irfft2`, covering top and bottom vertical modes, odd `201×201`, serialization, finite nonzero gradients, and an assertion that different Fourier modes have independent parameters.

```python
y = layer(x)
reference_fft[..., :my, :mx] = torch.einsum(
    "bixy,ioxy->boxy", x_fft[..., :my, :mx], weight_top
)
reference_fft[..., -my:, :mx] = torch.einsum(
    "bixy,ioxy->boxy", x_fft[..., -my:, :mx], weight_bottom
)
reference = torch.fft.irfft2(reference_fft, s=x.shape[-2:])
torch.testing.assert_close(y, reference, rtol=2e-5, atol=2e-5)
```

- [ ] Run `pytest -q tests/grouped_ufno_mionet_v3/test_spectral.py`; confirm the missing implementation fails.
- [ ] Implement FP32 FFT and paired-real learnable weights (`[..., 2]` converted with `view_as_complex`), clipped retained modes, positive/negative vertical halves, low-rank channel projections, and exact output-shape restoration. Keep residual/norm/local convolution in a separate `ComplexSpectralResidualBlock` so the primitive test remains exact.
- [ ] Verify focused tests on CPU and, when available, CUDA. Run `python -m pytest -q tests/grouped_ufno_mionet_v3/test_spectral.py`.
- [ ] Commit: `feat(v3): add learned complex spectral operator`

## Task 4: Implement differentiable ray time and phase-aligned coordinates

**Files:**

- Create: `grouped_ufno_mionet_v3/model/travel_time.py`
- Create: `grouped_ufno_mionet_v3/model/features.py`
- Test: `tests/grouped_ufno_mionet_v3/test_travel_time.py`
- Test: `tests/grouped_ufno_mionet_v3/test_features.py`

- [ ] Write failing tests for uniform `T_ray=r/v`, heterogeneous velocity gradients, coordinate gradients, dense-cache equality, causality features, bounded Gabor features, and SIREN initialization limits.

```python
travel = straight_ray_travel_time(velocity, source_xy, query_xy, samples=12)
expected = torch.linalg.vector_norm(query_xy - source_xy[:, None], dim=-1) / 2000.0
torch.testing.assert_close(travel.seconds, expected, rtol=2e-4, atol=2e-6)
travel.seconds.sum().backward()
assert velocity.grad is not None and velocity.grad.abs().sum() > 0
```

- [ ] Run the focused tests and confirm they fail before implementation.
- [ ] Implement line samples with `grid_sample(align_corners=True)`, physical-to-grid normalization, slowness/path summaries, `tau=t-t0-T_ray`, phase `2πf0τ`, Fourier bands, causal cone, and a bounded Gaussian-windowed sinusoidal bank. Validate coordinate/domain shapes explicitly.
- [ ] Verify with `pytest -q tests/grouped_ufno_mionet_v3/test_travel_time.py tests/grouped_ufno_mionet_v3/test_features.py`.
- [ ] Commit: `feat(v3): add differentiable phase aligned travel features`

## Task 5: Build velocity and source encoders with position-aware tokens

**Files:**

- Create: `grouped_ufno_mionet_v3/model/medium.py`
- Create: `grouped_ufno_mionet_v3/model/source.py`
- Test: `tests/grouped_ufno_mionet_v3/test_branches.py`

- [ ] Write failing tests for encode-once medium state, multiscale shape contracts, token position dependence, source-position/frequency dependence, record-to-medium expansion, and nonzero gradients through every complex spectral weight.
- [ ] Run the branch test and confirm failures.
- [ ] Implement a width-64 complex U-FNO pyramid with ranks/modes from config, explicit `(x,z)` Fourier position channels, pooled medium tokens, global medium rank, and sampled local pyramid features. Implement an independent source MLP plus source-map encoder using `[xs,zs,f0,t0,amplitude]` without zeroing source location.
- [ ] Add `required_gradient_groups()` to expose medium spectral, medium local, token, and source parameter groups for audits.
- [ ] Verify `pytest -q tests/grouped_ufno_mionet_v3/test_branches.py` and assert the configured model-size range is reached after assembly, not by dead padding parameters.
- [ ] Commit: `feat(v3): add position aware medium and source branches`

## Task 6: Assemble four-input MIONet and arbitrary point queries

**Files:**

- Create: `grouped_ufno_mionet_v3/model/fusion.py`
- Create: `grouped_ufno_mionet_v3/model/operator.py`
- Test: `tests/grouped_ufno_mionet_v3/test_query_operator.py`

- [ ] Write failing tests for the explicit multiplicative branches, local residual attention, arbitrary off-grid `(x,z,t)`, batching multiple sources per encoded medium, chunked/unchunked equality, and gradient participation of velocity/source/travel/trunk/query groups.

```python
state = model.encode_medium(velocity)
full = model.query(state, sources, coords, record_to_medium, chunk_size=None)
chunked = model.query(state, sources, coords, record_to_medium, chunk_size=257)
torch.testing.assert_close(full, chunked, rtol=2e-5, atol=2e-6)
```

- [ ] Run the query test and confirm missing operator failures.
- [ ] Implement `Bv(v,y) * Bs(source) * Btau(v,source,y,t) * Trunk(y,t)` rank products, summed projection, V2-style direct `query + sampled-local + attended-token` residual, query masking, the exact top free-surface factor, and chunking. Return normalized pressure only; physical decoding is a separate explicit method.
- [ ] Verify focused tests and ensure no receiver tensor appears in signatures or state dictionaries.
- [ ] Commit: `feat(v3): add phase aligned multi input query operator`

## Task 7: Add arbitrary-time complete wavefield rendering

**Files:**

- Create: `grouped_ufno_mionet_v3/model/dense.py`
- Extend: `grouped_ufno_mionet_v3/model/operator.py`
- Test: `tests/grouped_ufno_mionet_v3/test_dense_renderer.py`

- [ ] Write failing tests for output `[records,times,201,201]`, arbitrary nonuniform time lists, source-specific outputs, time-block equivalence, dense/query shared-coordinate agreement before correction, exact 401-frame streaming order, and cached dense travel time.
- [ ] Run the dense test and confirm failures.
- [ ] Implement grid-vectorized shared fusion, cached medium/source/grid travel state, coarse rank-channel fields, V2 multiscale local/source-map concatenation, retained source/time FiLM scale-bias modulation, time-conditioned complex spectral residual correction, the same exact free-surface factor, one-channel projection, and iterator-based streaming. Only `dense_time_block` may shrink after one CUDA OOM.
- [ ] Verify `pytest -q tests/grouped_ufno_mionet_v3/test_dense_renderer.py`; run a CUDA `201×201` smoke when CUDA is available and record peak memory.
- [ ] Commit: `feat(v3): render complete arbitrary time wavefields`

## Task 8: Implement phase-sensitive objectives and diagnostics

**Files:**

- Create: `grouped_ufno_mionet_v3/losses.py`
- Create: `grouped_ufno_mionet_v3/training/__init__.py`
- Create: `grouped_ufno_mionet_v3/training/audit.py`
- Test: `tests/grouped_ufno_mionet_v3/test_losses.py`
- Test: `tests/grouped_ufno_mionet_v3/test_audit.py`

- [ ] Write failing analytical tests for per-frame relative L2, complex coefficient loss, masked unit-phase loss, empty-mask failure, spatial gradients, temporal differences, query/dense consistency, zero baseline, energy ratio, and radial centroid displacement.
- [ ] Run focused tests and confirm missing implementation failures.
- [ ] Implement every approved loss in FP32 with named unweighted and weighted logging. Phase masks derive only from target energy and must contain a configured minimum number of modes. Implement branch gradient audit and anomaly-family report rejection.
- [ ] Verify `pytest -q tests/grouped_ufno_mionet_v3/test_losses.py tests/grouped_ufno_mionet_v3/test_audit.py`.
- [ ] Commit: `feat(v3): add phase sensitive training objectives`

## Task 9: Add versioned checkpointing and deterministic trainer

**Files:**

- Create: `grouped_ufno_mionet_v3/training/checkpoint.py`
- Create: `grouped_ufno_mionet_v3/training/trainer.py`
- Create: `scripts/train_grouped_v3.py`
- Test: `tests/grouped_ufno_mionet_v3/test_checkpoint_and_trainer.py`

- [ ] Write failing tests for checkpoint format `phase_aligned_complex_fno_mionet_v3`, V2 rejection, manifest/config mismatch rejection, atomic epoch checkpoints, deterministic resume, nonfinite stops, gradient-group stops, plateau detection, and fresh-history full-batch L-BFGS entry.
- [ ] Run the test and confirm failures.
- [ ] Implement AdamW training with fixed seeds, balanced family/phase sampling, AMP restricted to eligible local layers, gradient scaling/clipping, per-epoch atomic checkpoint, structured JSONL metrics, and timing breakdown for data/GPU. Enter deterministic full-batch L-BFGS only after recorded AdamW plateau; never resume stale curvature history.
- [ ] Verify focused tests and run `python scripts/train_grouped_v3.py --config configs/grouped_v3/smoke.yaml --dry-run`.
- [ ] Commit: `feat(v3): add guarded deterministic training and checkpoints`

## Task 10: Run structural CPU/CUDA smoke and close numerical contracts

**Files:**

- Create: `scripts/smoke_grouped_v3.py`
- Create: `tests/grouped_ufno_mionet_v3/test_structural_smoke.py`
- Update: `configs/grouped_v3/smoke.yaml`

- [ ] Write the end-to-end synthetic uniform/layered test before wiring the smoke command. It must perform one optimizer step for query+dense paths and assert finite loss plus finite nonzero gradients for all required groups.
- [ ] Run focused test to expose integration defects, fix only root causes, and retain regression tests.
- [ ] Run `pytest -q tests/grouped_ufno_mionet_v3`.
- [ ] Run `python scripts/smoke_grouped_v3.py --config configs/grouped_v3/smoke.yaml --device cpu`, then CUDA if available. Save `artifacts/grouped_ufno_mionet_v3/smoke/report.json` and do not commit runtime artifacts.
- [ ] Commit: `test(v3): verify structural wave operator smoke`

## Task 11: Execute the one-record uniform accuracy gate

**Files:**

- Create: `configs/grouped_v3/one_record_gate.yaml`
- Create: `scripts/overfit_grouped_v3.py`
- Create: `scripts/evaluate_grouped_v3.py`
- Test: `tests/grouped_ufno_mionet_v3/test_gate_policy.py`

- [ ] Write failing policy tests that block the nine-record phase unless early, middle, late, query, and centroid metrics all pass their independent thresholds.
- [ ] Implement deterministic energetic-record selection and atomic gate report/figures. The gate report includes exact record ID, config/manifest/checkpoint digests, time partitions, errors, centroid validity, gradients, throughput, and peak memory.
- [ ] Run the policy tests, then launch the bounded one-record job with the repository's current CUDA environment and log path under `artifacts/grouped_ufno_mionet_v3/one_record_gate/`.
- [ ] Train AdamW to plateau, optionally run fresh deterministic L-BFGS, evaluate the best checkpoint, and compare to the zero prediction.
- [ ] Stop and diagnose if any of these fail: early `<0.05`, middle `<0.05`, late `<0.05`, query `<0.05`, centroid `<1` spatial grid cell, all gradient groups present. Do not start Task 12 on failure.
- [ ] Commit code/config/report metadata only after the policy and command are reproducible: `feat(v3): add strict one record phase gate`.

## Task 12: Execute the balanced nine-record gate and enforce the pilot boundary

**Files:**

- Create: `configs/grouped_v3/nine_record_gate.yaml`
- Extend: `scripts/overfit_grouped_v3.py`
- Extend: `scripts/evaluate_grouped_v3.py`
- Test: `tests/grouped_ufno_mionet_v3/test_gate_policy.py`

- [ ] Add failing tests for exactly three energetic records per allowed family, no anomaly records, aggregate/per-family/late thresholds, exact-versus-interpolated separation, and pilot refusal when any metric fails.
- [ ] Implement balanced selection and the sealed policy result. A passed report is required input to any future pilot command.
- [ ] Run all V3 tests, launch the bounded nine-record job only if Task 11 passed, and evaluate its best checkpoint.
- [ ] Require aggregate query/dense `<0.10`, each family query/dense `<0.10`, each family late-four-frame `<0.10`, all values beat zero, and all required gradient groups are present.
- [ ] On success, write the final comparison figures and evidence report. On failure, write a diagnostic report and stop; do not relax data balance, losses, or gates.
- [ ] Commit: `feat(v3): seal balanced accuracy gate`

## Task 13: Final verification and handoff

**Files:**

- Update: `README.md`
- Create: `docs/superpowers/reports/2026-07-17-phase-aligned-complex-fno-mionet-v3-results.md`

- [ ] Run `pytest -q tests/grouped_ufno_mionet_v3` and the relevant unchanged V2 regression tests.
- [ ] Run `git diff --check`, inspect `git status --short`, and ensure no runtime artifact or source HDF5 is staged.
- [ ] Document exact commands, passed/failed gates, checkpoint/report paths, arbitrary-time full-field/query API examples, known limitations, and the explicit fact that FWI/receiver conditioning/anomaly training/production launch remain out of scope.
- [ ] Verify no GitHub remote mutation occurred. Keep all commits local unless the user separately authorizes a push.
- [ ] Commit: `docs(v3): record implementation and gate evidence`

## Mandatory stop conditions

- An `anomaly` record appears in any V3 batch or report.
- Manifest, normalization, checkpoint format, or split identity mismatches.
- Nonfinite targets, complex contractions, losses, or gradients.
- A required velocity/source/travel/trunk/query/dense/spectral gradient group is absent.
- Query and dense targets disagree at identical grid/time coordinates.
- The one-record gate fails: the nine-record gate must not begin.
- The nine-record gate fails: no pilot or production job may begin.
- One bounded CUDA OOM recovery may halve only `dense_time_block`; a second OOM stops the job.
