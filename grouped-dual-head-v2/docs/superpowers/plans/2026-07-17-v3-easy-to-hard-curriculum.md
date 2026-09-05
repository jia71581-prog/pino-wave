# V3 Easy-to-Hard Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the best balanced V3 pilot in the order uniform → layered → Marmousi slices, using deterministic replay and constrained checkpoint acceptance to improve difficult media without forgetting easier families.

**Architecture:** A sealed curriculum contract binds the completed pilot and defines three stages. A family-aware microbatch scheduler preserves grouped medium reuse and the single-source target contract; the trainer accumulates safe uniform microbatches, evaluates all families after every epoch, and promotes only checkpoints satisfying target improvement and forgetting limits.

**Tech Stack:** Python 3.13, PyTorch/CUDA, h5py VDS, multiprocessing DataLoader, NumPy, PyYAML, pytest, JSONL, atomic checkpoints, nohup/setsid.

---

### Task 1: Seal the curriculum parent and stage policy

**Files:**

- Create: `grouped_ufno_mionet_v3/training/curriculum.py`
- Create: `configs/grouped_v3/easy_to_hard_curriculum.yaml`
- Test: `tests/grouped_ufno_mionet_v3/test_curriculum_contract.py`

- [ ] **Step 1: Write failing parent-identity and acceptance tests**

```python
def test_curriculum_parent_requires_completed_pilot_and_best_checkpoint(tmp_path):
    identity = validate_curriculum_parent(
        terminal_report, best_validation, best_checkpoint,
        expected_manifest_digest="manifest", expected_run_digest="pilot-run",
    )
    assert identity.parent_epoch == 20 or identity.parent_epoch > 0
    assert identity.parent_checkpoint_sha256 == sha256(best_checkpoint)

def test_stage_acceptance_blocks_forgetting():
    decision = accept_stage_checkpoint(
        parent=family_scores(uniform=.50, layered=.70, marmousi=.55),
        candidate=family_scores(uniform=.54, layered=.65, marmousi=.56),
        target_family="layered", learned_families=("uniform",),
        aggregate_parent=2.30, aggregate_candidate=2.35,
    )
    assert decision.accepted
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q --import-mode=importlib tests/grouped_ufno_mionet_v3/test_curriculum_contract.py
```

Expected: collection fails because `grouped_ufno_mionet_v3.training.curriculum` does not exist.

- [ ] **Step 3: Implement immutable identity and acceptance rules**

Implement these public types and functions:

```python
@dataclass(frozen=True)
class CurriculumParentIdentity:
    manifest_digest: str
    pilot_run_digest: str
    parent_epoch: int
    parent_global_step: int
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    parent_validation_score: float

@dataclass(frozen=True)
class StageDecision:
    accepted: bool
    failures: Sequence[str]
```

Add `validate_curriculum_parent`, `accept_stage_checkpoint`, and `curriculum_run_digest` as public
functions beside these types. Reject a missing/incomplete terminal report, non-V3 checkpoint, manifest/run mismatch,
checkpoint path other than the recorded pilot best, nonfinite score, and SHA mismatch.

- [ ] **Step 4: Add the three-stage configuration**

The YAML must encode `uniform:2`, `layered:3`, `marmousi:5` epochs, learning rate `2e-5`,
weight decay `1e-6`, exact/interpolated fraction `0.25`, workers `8`, prefetch `4`, seed `29`,
aggregate tolerance `0.05`, replay tolerance `0.10`, and artifact root
`/data/jiayh/v3_easy_to_hard_curriculum`.

- [ ] **Step 5: Verify and commit**

Run the focused test and `git diff --check`, then commit:

```bash
git add grouped_ufno_mionet_v3/training/curriculum.py configs/grouped_v3/easy_to_hard_curriculum.yaml tests/grouped_ufno_mionet_v3/test_curriculum_contract.py
git commit -m "feat(v3): seal easy-to-hard curriculum"
```

### Task 2: Build deterministic family and replay microbatches

**Files:**

- Create: `grouped_ufno_mionet_v3/data/curriculum.py`
- Test: `tests/grouped_ufno_mionet_v3/test_curriculum_data.py`

- [ ] **Step 1: Write failing schedule tests**

```python
def test_uniform_step_is_three_four_record_microbatches(manifest):
    step = build_curriculum_schedule(manifest, stage="uniform", optimizer_steps=1, seed=29)[0]
    assert [len(m.record_indices) for m in step.microbatches] == [4, 4, 4]
    assert all(set(m.families) == {"uniform"} for m in step.microbatches)

def test_layered_and_marmousi_steps_preserve_groups_and_replay(manifest):
    layered = build_curriculum_schedule(manifest, stage="layered", optimizer_steps=4, seed=29)
    assert any(m.replay for m in layered[3].microbatches)
    marmousi = build_curriculum_schedule(manifest, stage="marmousi", optimizer_steps=5, seed=29)
    assert {m.family for s in marmousi for m in s.microbatches if m.replay} == {"uniform", "layered"}
```

Also assert one source per record, no duplicate sample IDs inside a microbatch, 25% strict midpoint
targets, grouped `record_to_medium`, and exactly one `read_wavefield` call per record.

- [ ] **Step 2: Run the focused test and verify RED**

Run the curriculum data test; expected failure is the missing module.

- [ ] **Step 3: Implement schedule dataclasses and family cycling**

```python
@dataclass(frozen=True)
class CurriculumMicrobatchSpec:
    record_indices: Sequence[int]
    family: str
    replay: bool
    loss_scale: float

@dataclass(frozen=True)
class CurriculumStepSpec:
    optimizer_step: int
    stage: str
    microbatches: Sequence[CurriculumMicrobatchSpec]
```

Uniform steps contain three four-record microbatches. Layered dominant steps contain three complete
four-source groups; every fourth step replaces the dominant batch with three uniform replay
microbatches. Marmousi dominant steps contain two five-source groups with loss scale `1.2`; a
five-step cycle includes one accumulated uniform replay step and one 12-record layered replay step.

- [ ] **Step 4: Implement lazy target reads and prefetch**

Reuse the pilot active-time selector, query sampler, `pack_v3_groups`, and `PilotBatch` tensor
contract. `CurriculumStepDataset` returns a tuple of fully materialized microbatches for one
optimizer step. `make_curriculum_loader` uses pinned memory, persistent workers, and the configured
prefetch factor.

- [ ] **Step 5: Verify and commit**

Run focused tests and commit:

```bash
git add grouped_ufno_mionet_v3/data/curriculum.py tests/grouped_ufno_mionet_v3/test_curriculum_data.py
git commit -m "feat(v3): schedule curriculum replay batches"
```

### Task 3: Train stages with accumulation and constrained promotion

**Files:**

- Create: `scripts/train_grouped_v3_curriculum.py`
- Test: `tests/grouped_ufno_mionet_v3/test_curriculum_training.py`

- [ ] **Step 1: Write failing trainer tests**

```python
def test_accumulation_performs_one_optimizer_update_for_three_microbatches():
    report = train_curriculum_step(trainer, microbatches, loss_fn)
    assert trainer.global_step == 1
    assert report["effective_record_count"] == 12

def test_stage_transition_uses_only_accepted_best_or_parent():
    assert select_stage_output(rejected_report) == rejected_report["parent_checkpoint"]
    assert select_stage_output(accepted_report) == accepted_report["best_checkpoint"]
```

Also test exact/interpolated validation keys, stage/family counts, checkpoint-per-epoch, run-digest
resume refusal, and rejection of nonfinite gradients.

- [ ] **Step 2: Run the focused test and verify RED**

Run the curriculum training test; expected failure is the missing script.

- [ ] **Step 3: Implement one optimizer update across microbatches**

For each optimizer step, zero gradients once, run `pilot_forward_loss` on every microbatch, scale
each loss by its configured `loss_scale / number_of_microbatches`, call backward, audit every
required gradient group, clip to `1.0`, and call AdamW `step()` once. Synchronize CUDA only around
measured regions. Report actual target/replay family counts and effective records.

- [ ] **Step 4: Implement stage lifecycle**

Load the sealed baseline parent with a fresh optimizer. At each stage, rebuild the optimizer at
`2e-5`, train its configured epochs, evaluate the fixed balanced exact and midpoint batches, save
`checkpoint_epoch_NNNN.pt`, and call `accept_stage_checkpoint`. Hardlink accepted stage best files;
when a stage has no acceptance, restore its unchanged parent before starting the next stage.

- [ ] **Step 5: Implement restart and terminal reports**

Bind every checkpoint to the curriculum run digest and stage name. Resume only at an epoch
boundary with matching identity. Atomically write `run_identity.json`, `metrics.jsonl`,
`validation_latest.json`, `stage_report.json`, `latest.pt`, `best.pt`, and final
`terminal_report.json`.

- [ ] **Step 6: Verify and commit**

Run focused tests and the complete V3 test directory, then commit:

```bash
git add scripts/train_grouped_v3_curriculum.py tests/grouped_ufno_mionet_v3/test_curriculum_training.py
git commit -m "feat(v3): train guarded easy-to-hard curriculum"
```

### Task 4: Benchmark, launch, monitor, and compare

**Files:**

- Create: `configs/grouped_v3/easy_to_hard_curriculum_benchmark.yaml`
- Create: `scripts/benchmark_grouped_v3_curriculum.py`
- Modify: `scripts/evaluate_grouped_v3_pilot.py`
- Create: `docs/superpowers/reports/2026-07-17-v3-easy-to-hard-curriculum-results.md`
- Test: `tests/grouped_ufno_mionet_v3/test_curriculum_benchmark.py`

- [ ] **Step 1: Write the failing benchmark-policy test**

Require 30 finite optimizer steps, complete gradient groups, nonzero exact/interpolated targets,
all required target/replay families, peak memory below device capacity, steady data wait below
20%, and positive effective records per second.

- [ ] **Step 2: Run and verify RED, then implement the benchmark wrapper**

The benchmark uses the production curriculum trainer for 10 steps per stage and writes an atomic
`benchmark_report.json`. It refuses launch if any policy gate fails.

- [ ] **Step 3: Run the real CUDA benchmark**

Write artifacts under `/data/jiayh/v3_easy_to_hard_curriculum_benchmark`. Sample `nvidia-smi` for
utilization, power, and memory. Preserve the model and losses; only workers, prefetch, and safe
microbatch sizes may change.

- [ ] **Step 4: Launch with nohup/setsid and monitor every epoch**

Use unbuffered Python, no sudo, PID/log files under the curriculum artifact root, and require PID
parent 1. At each epoch check finite losses, family/replay counts, checkpoint existence, GPU power,
data wait, disk space, validation acceptance, and stage transition provenance.

- [ ] **Step 5: Evaluate the accepted final model and baseline**

Run `scripts/evaluate_grouped_v3_pilot.py` on both checkpoints into distinct directories. Compare
held-out uniform/layered/Marmousi exact and midpoint 201×201 fields, off-grid point traces,
query/dense consistency, and inference timings. Retain the baseline if curriculum acceptance fails.

- [ ] **Step 6: Final verification and documentation**

Run the V3 and V2 suites separately, `git diff --check`, SHA-256 checks for checkpoint/report files,
and verify that no runtime artifacts are staged. Record negative results as well as improvements,
then commit:

```bash
git add configs/grouped_v3/easy_to_hard_curriculum_benchmark.yaml scripts/benchmark_grouped_v3_curriculum.py scripts/evaluate_grouped_v3_pilot.py tests/grouped_ufno_mionet_v3/test_curriculum_benchmark.py docs/superpowers/reports/2026-07-17-v3-easy-to-hard-curriculum-results.md README.md
git commit -m "docs(v3): record easy-to-hard curriculum evidence"
```

Do not implement FWI and do not push GitHub.
