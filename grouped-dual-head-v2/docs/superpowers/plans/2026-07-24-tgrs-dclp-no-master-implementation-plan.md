# DCLP-NO TGRS Submission Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified IEEE TGRS submission package for the acoustic DCLP-NO method, including leakage-free two-snapshot LoRA, strong baselines, direct dispersion evidence, publication-grade figures, and a compiled manuscript.

**Architecture:** Work is divided into four independently testable plans. The first freezes the data and evaluation contracts, the second implements two-snapshot LoRA without future access, the third builds the comparison and dispersion evidence, and the fourth turns sealed artifacts into the paper and submission package.

**Tech Stack:** Python 3.13, PyTorch, NumPy, SciPy, h5py, pandas, Matplotlib, pytest, YAML, IEEEtran LaTeX, BibTeX.

---

## Non-Git Checkpoint Rule

This continuation workspace has no `.git` directory. Every “checkpoint” step in
the subplans therefore means:

```bash
/home/jiayh/miniconda3/bin/python scripts/audit_tgrs_dclp_no.py \
  --config configs/tgrs_dclp_no/protocol.yaml \
  --source-root . \
  --output artifacts/tgrs_dclp_no/audit/source_manifest.json
```

Expected: exit code 0, `source_manifest.json` contains a deterministic SHA-256
for every registered source and configuration file, and no file under
`project_pristine` or `pretraining/gate4_long` changes.

If repository metadata is restored later, add normal Git commits after each
checkpoint without changing the scientific artifact identities.

## Execution Order

### Task 1: Freeze protocols and provenance

**Plan:** `docs/superpowers/plans/2026-07-24-tgrs-dclp-no-evidence-contract-plan.md`

- [ ] Execute every task in the evidence-contract plan.
- [ ] Require the final audit to report split counts `2240/480/480`.
- [ ] Require receiver indices to resolve to `z_index=2` and
  `x_index=10,15,...,190`.
- [ ] Require every observation pair to match nearest saved times to
  \(t_0+2/f_0\) and \(t_0+4/f_0\).
- [ ] Stop if the parent checkpoint SHA-256 is not
  `0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f`.

### Task 2: Implement and seal two-snapshot LoRA

**Plan:** `docs/superpowers/plans/2026-07-24-tgrs-dclp-no-two-snapshot-lora-plan.md`

- [ ] Execute the LoRA unit tests before any GPU run.
- [ ] Prove zero-initialized LoRA is numerically identical to the frozen parent.
- [ ] Prove only two non-adjacent registered snapshots are readable.
- [ ] Prove mutating every future frame leaves the sealed adapter digest
  unchanged.
- [ ] Run CPU smoke, one-record GPU smoke, three-family validation pilot, then
  freeze rank, layers, learning rate, steps, loss weights, and rollback gates.
- [ ] Run the complete test-ID adaptation only after the validation protocol is
  frozen.

### Task 3: Build baselines and direct dispersion evidence

**Plan:** `docs/superpowers/plans/2026-07-24-tgrs-dclp-no-dispersion-baselines-plan.md`

- [ ] Reproduce analytic FD2, FD4, and LWC-84 phase-velocity curves.
- [ ] Finish the parameter-matched Patch-DeepONet entry gates before its
  40-epoch run.
- [ ] Evaluate the global coarse field, frozen DCLP-NO, LoRA DCLP-NO, and every
  accepted baseline with the same 480-record manifest.
- [ ] Compute full-field, wavefront, spectral, and 37-receiver metrics.
- [ ] Compute paired medium-group bootstrap intervals and Holm-corrected claim
  decisions.
- [ ] Materialize `claim_gate.json`; manuscript generation must read this file
  instead of deciding claims manually.

### Task 4: Generate the TGRS paper and submission package

**Plan:** `docs/superpowers/plans/2026-07-24-tgrs-dclp-no-paper-package-plan.md`

- [ ] Verify primary-source references before adding them to
  `references.bib`.
- [ ] Generate the graphical abstract and Figs. 1–7 from registered artifacts.
- [ ] Write tables from machine-readable result files.
- [ ] Draft the paper in the approved section order.
- [ ] Make title and claim language conditional on `claim_gate.json`.
- [ ] Compile IEEEtran PDF, inspect every page, verify fonts/graphics/citations,
  and write the final SHA-256 submission manifest.

## Global Stop Gates

- [ ] Do not start a GPU job while the RTX 3090 lacks sufficient free memory.
- [ ] Do not treat a launched process as successful until PID, advancing log,
  GPU utilization, and checkpoint output are all verified.
- [ ] Do not use receiver traces or \(t>t_b\) wavefields during adaptation.
- [ ] Do not use incomplete, zero-update, or collapsed DeepONet attempts as the
  strongest published baseline.
- [ ] Do not describe LWC-84 or DCLP-NO as dispersion free.
- [ ] Do not include elastic-wave results or language.
- [ ] Do not finalize the title before the claim-gate audit.

## Final Acceptance Command

```bash
/home/jiayh/miniconda3/bin/python scripts/verify_tgrs_submission.py \
  --paper-dir paper/tgrs_dclp_no \
  --artifact-dir artifacts/tgrs_dclp_no \
  --require-figures 7 \
  --require-receiver-count 37 \
  --require-observed-snapshots 2
```

Expected: exit code 0 and
`paper/tgrs_dclp_no/submission_manifest.json` reports:

```json
{
  "status": "verified",
  "acoustic_only": true,
  "observed_snapshot_count": 2,
  "receiver_count": 37,
  "future_truth_leakage": false,
  "unresolved_markers": 0,
  "missing_citations": 0,
  "missing_figures": 0,
  "pdf_page_visual_checks_passed": true
}
```
