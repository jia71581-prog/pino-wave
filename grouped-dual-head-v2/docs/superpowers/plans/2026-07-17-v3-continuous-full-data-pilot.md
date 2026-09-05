# V3 Continuous-Time Full-Data Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the passed phase-aligned Complex-FNO Multi-Input DeepONet V3 on all 2,240 allowed training records with exact and interpolated-time targets, validate exact and interpolated outputs separately, and produce a resumable pilot checkpoint without receiver conditioning or source superposition.

**Architecture:** A deterministic grouped schedule selects equal record counts from uniform, layered, and marmousi while packing all sources belonging to the same medium through `record_to_medium`. CPU worker processes read and interpolate physical VDS frames ahead of the GPU. The existing query and dense paths retain true-target losses, and a sealed nine-record passed report is mandatory before startup.

**Tech Stack:** Python 3.12, PyTorch DataLoader multiprocessing/pinned memory, h5py SWMR/VDS, NumPy, PyYAML, CUDA FP32, pytest, JSONL, atomic PyTorch checkpoints.

---

## Task 1: Add a sealed pilot launch contract

**Files:**

- Create: `grouped_ufno_mionet_v3/training/pilot.py`
- Create: `configs/grouped_v3/continuous_pilot.yaml`
- Test: `tests/grouped_ufno_mionet_v3/test_pilot_contract.py`

- [ ] Write a failing test that accepts only a passed nine-record report with the active manifest, exactly `3/3/3` gate records, every family metric below `0.10`, and a checkpoint whose SHA-256 matches the recorded parent identity.

```python
identity = validate_pilot_prerequisite(
    passed_report,
    checkpoint,
    expected_manifest_digest="manifest",
)
assert identity.gate_step == 750
assert identity.parent_checkpoint_sha256 == sha256(checkpoint)
```

- [ ] Add failure cases for a failed decision, anomaly family, wrong counts, metric `>=0.10`, manifest mismatch, missing checkpoint, and empty gradient groups.
- [ ] Run `pytest -q tests/grouped_ufno_mionet_v3/test_pilot_contract.py`; confirm the missing module fails.
- [ ] Implement `PilotIdentity` and `validate_pilot_prerequisite` with no permissive fallback. The returned identity contains gate report/checkpoint absolute paths and SHA-256 values for checkpoint provenance.
- [ ] Add immutable pilot configuration values: `batch_records=12`, `continuous_fraction=0.25`, `workers=8`, `prefetch_factor=4`, `epochs=20`, `steps_per_epoch=187`, checkpoint root under `/data/jiayh/v3_continuous_pilot`, and deterministic seed 17.
- [ ] Verify the focused test and config load, then commit `feat(v3): seal continuous pilot launch contract`.

## Task 2: Build deterministic grouped exact/interpolated CPU prefetch

**Files:**

- Create: `grouped_ufno_mionet_v3/data/pilot.py`
- Test: `tests/grouped_ufno_mionet_v3/test_pilot_data.py`

- [ ] Write synthetic-HDF5 tests for exactly four records per family, no duplicate sample IDs in a step, medium reuse through `record_to_medium`, one source per target, three early/middle/late requested times, and a deterministic exact/interpolated mask matching `continuous_fraction`.
- [ ] Test that interpolated targets equal adjacent physical frames, query targets are sampled from the same dense target tensor, and importance probabilities are finite and positive.
- [ ] Run the focused test and confirm the missing implementation fails.
- [ ] Implement a precomputed `PilotStepSpec` schedule. Each step selects four uniform records, one complete four-source layered group, and four of the five sources in one marmousi group. Cycle omitted marmousi sources deterministically so every record is covered.
- [ ] Implement `PilotBatchDataset.__getitem__` using process-local `V3WavefieldDataset` handles. For each record, select one early, one middle, and one late time; convert a deterministic fraction to strict between-frame times; read targets using `read_wavefield`; sample query grid locations from target energy plus a uniform floor; and return CPU tensors plus exact/interpolated flags.
- [ ] Implement a top-level identity collator compatible with multiprocessing and a `make_pilot_loader` function using pinned memory, persistent workers, and configured prefetch.
- [ ] Verify the focused test, then commit `feat(v3): prefetch grouped continuous wavefield batches`.

## Task 3: Implement resumable full-data training and split validation

**Files:**

- Create: `scripts/train_grouped_v3_pilot.py`
- Extend: `grouped_ufno_mionet_v3/training/pilot.py`
- Test: `tests/grouped_ufno_mionet_v3/test_pilot_training.py`

- [ ] Write failing tests proving the CLI refuses startup without the sealed report, binds the manifest/config/parent digests, restores an epoch checkpoint deterministically, writes one checkpoint per epoch, and reports exact and interpolated metrics under different keys.
- [ ] Test that query and dense losses consume targets from each record's single source only and that `record_to_medium` is passed unchanged into `prepare_sources`.
- [ ] Run the focused test and confirm failure before implementation.
- [ ] Implement initialization from the sealed nine-record model state with a fresh AdamW optimizer. Resume only from a pilot checkpoint with matching manifest and pilot run digest.
- [ ] Implement the GPU step with grouped medium encoding, three dense frames per record, adaptive query importance sampling, exact free-surface behavior, existing phase-sensitive losses, finite-gradient audits, clipping, and structured timing for CPU wait, host-to-device transfer, forward/backward, and checkpoint I/O.
- [ ] Implement fixed deterministic validation records from all three families. Report exact saved-frame query/dense/late relative L2 separately from midpoint-interpolated query/dense relative L2. Validation observations never enter training.
- [ ] Write atomic `latest.pt`, `checkpoint_epoch_NNNN.pt`, `metrics.jsonl`, `run_identity.json`, and `validation_latest.json` under the configured `/data` artifact root.
- [ ] Verify the focused test and full V3 test suite, then commit `feat(v3): train resumable continuous full-data pilot`.

## Task 4: Benchmark the data/GPU pipeline and tune only safe throughput controls

**Files:**

- Create: `configs/grouped_v3/continuous_pilot_smoke.yaml`
- Create: `scripts/benchmark_grouped_v3_pilot.py`
- Test: `tests/grouped_ufno_mionet_v3/test_pilot_benchmark.py`

- [ ] Write a failing benchmark-policy test requiring finite loss, all gradient groups, nonzero interpolated records, peak CUDA memory below device capacity, and measured CPU-wait/step timings.
- [ ] Implement a three-step dry benchmark that uses the production loader/model/loss path and writes `benchmark_report.json` atomically.
- [ ] Run the benchmark with 8 workers, prefetch 4, pinned memory, and batch 12. Sample `nvidia-smi` utilization/power during the steady steps.
- [ ] If peak memory is below 70% and CPU wait is below 20% of step time, try batch 24 once. Keep the larger batch only if it has no OOM, preserves family balance, and improves records/second by at least 10%. On OOM, return to batch 12; do not shrink model modes or remove losses.
- [ ] If CPU wait exceeds 20%, raise prefetch to 6 or workers to at most 12, bounded by the detected 10 physical CPU cores and 120 GB available RAM. Do not copy the 167 GB source dataset.
- [ ] Verify the benchmark policy and commit `perf(v3): validate continuous pilot throughput`.

## Task 5: Launch, monitor, validate, and hand off the pilot

**Files:**

- Create: `docs/superpowers/reports/2026-07-17-v3-continuous-pilot-results.md`
- Update: `README.md`

- [ ] Launch with `nohup` and unbuffered Python, writing PID and combined stdout/stderr under `/data/jiayh/v3_continuous_pilot/logs/`. Do not use sudo.
- [ ] Monitor the log, process identity, epoch checkpoint, GPU utilization/power, disk free space, and validation metrics. A missing process without a terminal report is a failure, not completion.
- [ ] After every epoch, require finite training loss, no missing gradients, all three families present, a nonzero interpolated fraction, and an atomic checkpoint. Stop on a repeated OOM, identity mismatch, anomaly record, or nonfinite value.
- [ ] Select the best checkpoint by exact and interpolated validation aggregate score without using receiver diagnostics. Generate representative uniform/layered/marmousi exact-frame wavefield comparisons and continuous-time point-query diagnostics.
- [ ] Run V3 and V2 regression suites separately, `git diff --check`, checkpoint/report SHA-256 verification, and confirm no runtime artifact is staged.
- [ ] Document throughput, peak memory, exact/interpolated validation metrics, checkpoint/log paths, limitations, and whether the pilot is accurate enough to justify a longer production run. Do not implement FWI or push GitHub.
- [ ] Commit `docs(v3): record continuous pilot evidence`.

## Mandatory stop conditions

- The sealed nine-record report or checkpoint provenance fails validation.
- Any `anomaly` record appears in a batch, validation report, or checkpoint identity.
- A record target contains more than one source or grouped sources are numerically superposed.
- Manifest, normalization, checkpoint format, or pilot run digest mismatches.
- Targets, predictions, losses, or required gradients become nonfinite.
- Query and dense targets disagree at identical grid/time coordinates.
- Two CUDA OOM events occur after returning to batch 12.
- Available artifact-disk space falls below 20 GB.
