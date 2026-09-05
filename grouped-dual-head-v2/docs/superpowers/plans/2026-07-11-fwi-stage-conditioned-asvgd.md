# Stage-Conditioned FWI and L2-Guarded ASVGD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible stage-wise gradient conditioning to deterministic FWI and prevent low-rank ASVGD output from replacing a demonstrably better L2 result.

**Architecture:** Preserve the frozen Python 3.13 FWI runner.  Add pure, independently tested conditioning functions in `scripts/fwi_conditioning.py`; then let `run_asvgd_bottom_receivers.py` install a custom Adam implementation only for the first (L2) optimizer construction.  A post-run guard reads the runner's artifacts and writes a separate guarded result/figure when the ASVGD receiver NMSE exceeds the L2 baseline by a configured tolerance.

**Tech Stack:** Python 3.13, PyTorch, NumPy, SciPy-compatible geometry, Deepwave through the preserved runner, pytest.

---

### Task 1: Add pure stage-conditioning primitives

**Files:**

- Create: `scripts/fwi_conditioning.py`
- Create: `tests/test_fwi_conditioning.py`

- [x] **Step 1: Write failing unit tests for schedules, Gaussian gradient smoothing, ray illumination, Huber-TV, and guard selection.**

```python
def test_parse_stage_schedule_broadcasts_one_value_and_rejects_wrong_count() -> None:
    assert parse_stage_schedule("2", stages=4, option="sigma") == [2.0] * 4
    with pytest.raises(ValueError, match="four values"):
        parse_stage_schedule("1,2", stages=4, option="sigma")

def test_gaussian_gradient_smoothing_preserves_constant_and_reduces_impulse() -> None:
    constant = torch.ones((9, 9))
    impulse = torch.zeros((9, 9)); impulse[4, 4] = 1.0
    assert torch.allclose(gaussian_smooth_2d(constant, sigma_cells=1.5), constant)
    assert gaussian_smooth_2d(impulse, sigma_cells=1.5)[4, 4] < 1.0

def test_ray_illumination_gain_is_bounded_and_deemphasises_well_covered_cells() -> None:
    illumination = ray_illumination(9, 9, np.array([[0, 4]]), np.array([[8, 4]]))
    gain = illumination_gain(illumination, max_gain=3.0)
    assert float(gain.min()) >= 1.0 / 3.0
    assert float(gain.max()) <= 3.0
    assert gain[4, 4] < gain[0, 0]

def test_huber_tv_on_slowness_is_zero_for_constant_velocity_and_positive_at_an_edge() -> None:
    constant = torch.full((8, 8), 3000.0)
    edge = constant.clone(); edge[:, 4:] = 4000.0
    assert huber_tv_slowness_squared(constant, reference_velocity_mps=3000.0, delta=1e-3) == pytest.approx(0.0)
    assert huber_tv_slowness_squared(edge, reference_velocity_mps=3000.0, delta=1e-3) > 0.0

def test_select_l2_guard_falls_back_only_when_asvgd_exceeds_tolerance() -> None:
    assert select_l2_guard(l2_nmse=0.1, candidate_nmse=0.101, tolerance=0.02).selected == "asvgd"
    assert select_l2_guard(l2_nmse=0.1, candidate_nmse=0.103, tolerance=0.02).selected == "l2"
```

- [x] **Step 2: Run the focused tests and verify they fail because `scripts.fwi_conditioning` does not exist.**

Run: `PATH=/home/jiayh/miniconda3/bin:$PATH pytest -q tests/test_fwi_conditioning.py`

Expected: collection error naming `scripts.fwi_conditioning`.

- [x] **Step 3: Implement only the tested helpers.**

```python
def parse_stage_schedule(text: str, *, stages: int, option: str) -> list[float]: ...
def gaussian_smooth_2d(field: torch.Tensor, *, sigma_cells: float) -> torch.Tensor: ...
def ray_illumination(nz: int, nx: int, source_ij: np.ndarray, receiver_ij: np.ndarray) -> torch.Tensor: ...
def illumination_gain(illumination: torch.Tensor, *, max_gain: float) -> torch.Tensor: ...
def huber_tv_slowness_squared(velocity: torch.Tensor, *, reference_velocity_mps: float, delta: float) -> torch.Tensor: ...
def select_l2_guard(*, l2_nmse: float, candidate_nmse: float, tolerance: float) -> GuardDecision: ...
```

The Gaussian operation applies to a gradient, never directly to the velocity.  Ray illumination is a static, clipped approximation used only as an optional preconditioner; it must not claim to be a full Hessian.

- [x] **Step 4: Run the focused tests and verify they pass.**

Run: `PATH=/home/jiayh/miniconda3/bin:$PATH pytest -q tests/test_fwi_conditioning.py`

Expected: all tests pass.

### Task 2: Install stage-conditioned L2 Adam without changing MAP/ASVGD

**Files:**

- Modify: `scripts/run_asvgd_bottom_receivers.py`
- Modify: `tests/test_asvgd_bottom_receivers.py`

- [x] **Step 1: Write failing tests for the new CLI knobs and first-call-only optimizer factory.**

```python
def test_wrapper_help_lists_stage_conditioning_options() -> None:
    completed = subprocess.run([... "--bottom-receivers", "2", "--help"], ...)
    assert "--l2-gradient-sigma-cells" in completed.stdout
    assert "--posterior-l2-guard-relative-nmse" in completed.stdout

def test_l2_optimizer_factory_conditions_only_the_first_lbfgs_construction() -> None:
    factory = make_l2_optimizer_factory(original_lbfgs=FakeLBFGS, conditioning=FakeConditioning())
    assert isinstance(factory([torch.nn.Parameter(torch.zeros(()))], lr=0.1), StageConditionedAdam)
    assert isinstance(factory([torch.nn.Parameter(torch.zeros(()))], lr=0.1), FakeLBFGS)
```

- [x] **Step 2: Run those tests and verify they fail because the new factory/CLI options are absent.**

Run: `PATH=/home/jiayh/miniconda3/bin:$PATH pytest -q tests/test_asvgd_bottom_receivers.py -k 'conditioning or factory'`

Expected: import/attribute failure naming the missing symbols.

- [x] **Step 3: Implement the optimizer factory and CLI.**

Add opt-in parameters:

```text
--l2-gradient-sigma-cells 10,6,3,1
--l2-huber-tv-weights 0,0,1e-4,5e-5
--l2-huber-tv-delta 1e-3
--l2-illumination-preconditioner {none,ray}
--l2-illumination-max-gain 3
--posterior-l2-guard-relative-nmse 0.02
```

Capture the padded source/receiver geometry in the existing geometry wrapper.  The first LBFGS construction returns an Adam whose `step(closure)` executes the closure once, then applies stage-selected Gaussian gradient smoothing, optional clipped ray-illumination gain, and the selected Huber-TV gradient before calling Adam.  Subsequent LBFGS calls delegate unchanged to the original factory, preserving MAP.  Empty schedules and `illumination=none` retain the old gradient behavior.

- [x] **Step 4: Verify wrapper tests and CLI.**

Run: `PATH=/home/jiayh/miniconda3/bin:$PATH pytest -q tests/test_asvgd_bottom_receivers.py tests/test_fwi_conditioning.py && PATH=/home/jiayh/miniconda3/bin:$PATH python scripts/run_asvgd_bottom_receivers.py --bottom-receivers 2 --help`

Expected: tests pass and help lists all six options.

### Task 3: Persist an L2-guarded posterior artifact

**Files:**

- Modify: `scripts/run_asvgd_bottom_receivers.py`
- Modify: `tests/test_asvgd_bottom_receivers.py`

- [x] **Step 1: Write a failing artifact test with a temporary summary and NPZ result.**

```python
def test_posterior_guard_writes_l2_fallback_for_worse_asvgd(tmp_path: Path) -> None:
    write_minimal_result(tmp_path, l2_nmse=0.01, asvgd_nmse=0.02)
    report = apply_posterior_l2_guard(tmp_path, tolerance=0.02, runner=FakeRunner())
    assert report["selected"] == "l2"
    assert (tmp_path / "posterior_l2_guard.json").is_file()
    assert np.allclose(np.load(tmp_path / "comparison_result_l2_guarded.npz")["selected_velocity"], l2_velocity)
```

- [x] **Step 2: Run the artifact test and verify it fails because `apply_posterior_l2_guard` is missing.**

Run: `PATH=/home/jiayh/miniconda3/bin:$PATH pytest -q tests/test_asvgd_bottom_receivers.py::test_posterior_guard_writes_l2_fallback_for_worse_asvgd`

Expected: collection failure naming `apply_posterior_l2_guard`.

- [x] **Step 3: Implement artifact guard and guarded figures.**

Read `summary.json` and `comparison_result.npz`, compare the selected ASVGD receiver NMSE to the L2 NMSE, and write `posterior_l2_guard.json` plus `comparison_result_l2_guarded.npz`.  If the candidate exceeds the tolerance, use `l2_velocity`/`l2_prediction` in the guarded NPZ and write separate comparison/gather figures labelled `L2 guard fallback`; otherwise retain the selected ASVGD arrays.  Preserve all raw arrays and existing files.

- [x] **Step 4: Invoke the guard after `runner.run(args)` and rerun all focused tests.**

Run: `PATH=/home/jiayh/miniconda3/bin:$PATH pytest -q tests/test_asvgd_bottom_receivers.py tests/test_fwi_conditioning.py`

Expected: all tests pass.

### Task 4: Run a controlled CPU smoke test and prepare the GPU configuration

**Files:**

- Create: `artifacts/marmousi_hdf5_fwi/stage_conditioned_cpu_smoke_20260711/`

- [x] **Step 1: Run a 101×101 synthetic anomaly smoke using all four source stages and the new options.**

Run the wrapper with a 101×101 circular anomaly, 2 shots, 21 top plus 4 bottom receivers, `nt=240`, 4 L2 iterations, zero MAP/ASVGD iterations, `--l2-optimizer adam`, `--l2-gradient-sigma-cells 4,3,2,1`, `--l2-huber-tv-weights 0,0,1e-4,5e-5`, and `--l2-illumination-preconditioner ray`.

- [x] **Step 2: Verify the smoke log reports all four stages, conditioned Adam, and a guard artifact.**

Run: `rg 'lowpass=(5|8|10|15)|Stage-conditioned Adam|posterior_l2_guard' <log>`.

Expected: four stage labels, no MAP optimizer substitution, and guard files.

### Task 5: Review and commit only task-scoped code

**Files:**

- Modify: `docs/superpowers/plans/2026-07-11-fwi-stage-conditioned-asvgd.md`
- Create: `scripts/fwi_conditioning.py`
- Modify: `scripts/run_asvgd_bottom_receivers.py`
- Create/Modify: `tests/test_fwi_conditioning.py`, `tests/test_asvgd_bottom_receivers.py`

- [x] **Step 1: Mark verified tasks complete and inspect the scoped diff.**

Run: `git diff --check -- scripts/fwi_conditioning.py scripts/run_asvgd_bottom_receivers.py tests/test_fwi_conditioning.py tests/test_asvgd_bottom_receivers.py docs/superpowers/plans/2026-07-11-fwi-stage-conditioned-asvgd.md`

Expected: no whitespace errors.

- [x] **Step 2: Commit only the listed code, tests, and plan.**

```bash
git add scripts/fwi_conditioning.py scripts/run_asvgd_bottom_receivers.py tests/test_fwi_conditioning.py tests/test_asvgd_bottom_receivers.py docs/superpowers/plans/2026-07-11-fwi-stage-conditioned-asvgd.md
git commit -m "feat: add stage-conditioned FWI safeguards"
```

Do not stage generated artifacts, existing user changes, or unrelated untracked files.  Do not push unless explicitly requested.

## Plan self-review

- Spec coverage: staged Gaussian-on-gradient, illumination compensation, Huber-TV, and L2-versus-ASVGD guard are each testable independently.
- Safety: the frozen runner stays immutable; only its first optimizer factory call is replaced, so MAP and ASVGD implementations are not silently altered.
- Validation: pure tensor tests, wrapper integration tests, and an all-stage CPU smoke precede any new GPU experiment.
