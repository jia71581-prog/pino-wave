# V4 Blockwise L-BFGS Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restart the best exact-time V4 checkpoint with deterministic head-only L-BFGS, effective batch 48, per-step validation, and atomic observability.

**Architecture:** A small optimizer-policy module validates trainable parameter selection and fixed-batch closure semantics. A dedicated runner reconstructs the bound deep-phase model, loads the V4 parent, freezes the backbone, caches four exact-time macros, executes strong-Wolfe L-BFGS outer steps, and applies the preregistered validation gate.

**Tech Stack:** Python 3.12, PyTorch 2.7, CUDA/RTX 3090, pytest, existing V4 exact-time data/model/metrics.

---

### Task 1: Define deterministic blockwise optimizer policy

**Files:**
- Create: `saved_time_phase_operator_v4/lbfgs.py`
- Create: `tests/saved_time_phase_operator_v4/test_lbfgs.py`

- [ ] Write tests proving only dense-decoder parameters remain trainable, effective batch equals macro records times accumulated macros, and repeated closures consume the same fixed batch identifiers.
- [ ] Run `python -m pytest -q tests/saved_time_phase_operator_v4/test_lbfgs.py` and verify it fails because the module is absent.
- [ ] Implement `freeze_for_dense_lbfgs`, `effective_batch_records`, and `FixedClosureBatch` with validation for empty/mutating batches.
- [ ] Re-run the focused tests and verify they pass.
- [ ] Commit as `feat(v4): define deterministic blockwise lbfgs policy`.

### Task 2: Implement the bound refinement runner

**Files:**
- Create: `scripts/refine_saved_time_v4_lbfgs.py`
- Create: `configs/saved_time_v4/lbfgs_refinement_batch48.yaml`
- Modify: `tests/saved_time_phase_operator_v4/test_lbfgs.py`

- [ ] Add failing tests for parent metric gates and family-regression rejection.
- [ ] Implement exact V4 parent loading, manifest/config verification, a four-macro cached closure, strong-Wolfe L-BFGS, exact validation, atomic step checkpoints, and JSONL logging.
- [ ] Run the focused tests and a CPU policy smoke.
- [ ] Commit as `feat(v4): add batch-48 lbfgs refinement runner`.

### Task 3: CUDA smoke, probe, and decision

**Files:**
- Create: `docs/superpowers/reports/2026-07-18-blockwise-lbfgs-v4-refinement-results.md`

- [ ] Run one CUDA outer step with `--smoke-steps 1`; require finite loss, at least two closure evaluations, and peak allocation below 23 GiB.
- [ ] Run the complete V3+V4 regression suite.
- [ ] Launch the 12-step refinement through detached `nohup`, and run the local per-step log monitor.
- [ ] Compare the best checkpoint with the bound parent using the 2% aggregate and 3% family gates.
- [ ] Record the outcome and commit the report. Authorize continuation only if both gates pass.
