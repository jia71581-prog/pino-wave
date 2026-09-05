# Coarse-201 (h=10 m) larger-dt rejection evidence — read-only review, 2026-09-05

Bounded read-only long-reasoning task from the 2026-09-05 handoff. No new GPU runs, no sealed truth reads
(only velocity_mps inputs, split/medium_type labels, and existing frozen JSON artifacts were read).

## Question
Before considering any micro residual head on R6: is there existing rejection evidence for running the
coarse 201x201 (h=10 m) LWC-84 parent at internal dt larger than the tested 125/250 us?

## Evidence chain

### 1. Frozen QC of the coarse pipeline: lwc_qmax < 1.0, enforced at runtime
- qmax definition (identical in solver metric and preflight): qmax = dt^2 * vmax^2 * (2048/315) * (1/dx^2 + 1/dz^2)
  (src/fno_acoustic/data_generation/solver_lwc84.py:382, solver_lwc84_fused.py:510-512;
   scripts/preflight_v23b_stability.py:41-47 with 1/dx^2+1/dz^2 = 0.02 for h=10 m).
- Coarse shard evaluator hard-aborts any batch with qmax >= 1.0
  (scripts/evaluate_coarse_lwc84_201_shard.py:219-222, FloatingPointError "unstable solver symbol").
- The 2026-09-05 A/B parent gate froze "A_and_B_cfl_qmax_strictly_below = 1.0"
  (results/accelerated_coarse_lwc84_residual_r1_20260905/parent_gate_preregistration.json, physics_QC.stability).
- Note: the R6 fine-grid (401, h=5 m) gate uses a different frozen limit LWC_QMAX_LIMIT = 9.6
  (scripts/gate_r6_anchored_r54_runtime.py:47) appropriate to its own configuration. The coarse-201 QC is
  qmax < 1 and may not be unilaterally relaxed.

### 2. Historical rejection at dt = 500 us (v23)
results/r16_dscp_v23b_coarse_lwc84_dispersion_preregistration_20260827.json, "supersedes" clause:
"The frozen 0.5 ms coarse step exceeded the LWC-84 spectral stability limit on all four first validation
batches; every worker aborted before producing a worker result, so v23 is an invalid numerical
configuration rather than an outcome-bearing audit." v23b re-ran at internal_dt_s = 0.000125.

### 3. Input-side qmax sweep over the CURRENT dataset (velocity inputs only; no truth)
Dataset: acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5, 4003 records, all splits.
Core families = uniform/layered/marmousi (3203 records; train core 2240).
- dt = 250 us: max qmax 0.515 (all records incl. anomaly) -> 0 violations. Consistent with arm B passing QC.
- dt = 500 us: core-family violations 690/3203 (train-only 449/2240), core max qmax 1.169;
  all-family 798/4003, max 2.061.
- dt = 625 us: core-family violations 1964/3203 (61%; train-only 1378/2240), core max qmax 1.827;
  all-family 2316/4003, max 3.221.
- Input-side dt ceiling under qmax < 1: core-family pool dt < 462.4 us (core vmax 5997.2 m/s);
  whole dataset incl. anomaly dt < 348.3 us (vmax 7962.9 m/s, a train anomaly record).
- cfl_2d is not the binding constraint (e.g. 625 us, v=5997 -> cfl 0.530 < 1); qmax binds first.
- Handoff's single-record example confirmed and strengthened: uniform B qmax 0.167486 -> 625 us extrapolation
  1.046789 >= 1; but the worst record in the 24-record A/B panel is layered train_layered_00660
  (qmax250 = 0.274552) -> already 1.098 >= 1 at 500 us and 1.716 at 625 us.

### 4. dt = 250 us (arm B) is accuracy-clean but runtime-rejected
results/accelerated_coarse_lwc84_residual_r1_20260905/parent_gate_report.json, status "parent_gate_rejected":
- Accuracy QC all passed: A vs truth 0.035898, B vs truth 0.036131 (energy aggregate), B-A aggregate 2.33e-4,
  max record delta 8.9e-4; phase/spectrum/centroid/temporal gates all True.
- Runtime gates failed: B_mean 1.4672 s > 0.9 s (B_runtime_mean_max), B_p95 1.660 s > 1.0 s (B_runtime_p95_max).
  B is also slower than the R6 fine parent itself (~0.8897 s exposed-train recheck mean; R6-anchored
  device runtime 1.287 s including the zero head).

### 5. The old dt = 500 us "success" is not usable
results/tgrs_coarse_ladder_dt5.0e-4/summary.json: aggregate rel-L2 0.925%, but on the OLD dataset
(acoustic_lwc84_2km_401x401_to_201_v1, teacher dt 125 us), only 3 validation records, development_gate_only.
Old data + validation records: may not be used for parameter selection nor to reopen truth. Its dataset
predates the current velocity pool (no qmax>=1 media). tgrs_coarse_ladder_dt1.0e-3 and dt2.0e-3 directories
are EMPTY (created 2026-08-05, no summary) — those ladder rungs never produced results.

### 6. Dispersion context (existing CPU symbolic result, not new)
tgrs_dclp_no/dispersion.py: at 50 Hz, axial phase-velocity lag is 179.002/181.073 ppm for (h5, dt125/625 us)
vs 22112.533/22112.578 ppm for (h10, dt125/250 us). Spatial coarsening to h=10 m dominates dispersion
(~2.2% at 50 Hz) and is essentially dt-independent; coarse parent truth error ~3.6% vs R6 fine parent
0.26-0.29%. Same-grid residual correction on R6 is NOT coarse-space dedispersion; a coarse-201 residual
head would have to correct a ~14x larger parent error.

## Conclusions (accept/reject with reasons)
- dt = 625 us on coarse-201: REJECTED without any run — violates the frozen qmax < 1 QC on 61% of
  core-family records (input-side, max qmax 1.827). Relaxing the QC is out of scope and forbidden.
- dt = 500 us on coarse-201: REJECTED — historical v23 abort (all first validation batches) plus input-side
  recomputation (690/3203 core records qmax >= 1). The old ladder "0.925%" is old-data/validation evidence
  and cannot rehabilitate it.
- dt = 250 us on coarse-201: accuracy-viable, runtime-rejected (1.4672 s mean vs 0.9 s gate; slower than the
  R6 fine parent). No speed motivation for a residual head on this parent.
- Only untested headroom under the existing QC: 250 us < dt < ~462 us (core pool input-side bound, e.g. a
  312.5 us = dt/2-of-625 ladder rung, worst-case core qmax 0.457). But arm B at 250 us already fails the
  runtime target, and internal steps scale ~1/dt, so even ~462 us would give only ~1.85x fewer steps than
  250 us: projected mean ~0.8 s at best — marginal vs R6 0.89 s with a 14x worse error floor. Not worth a
  GPU run for the current goal.

## Recommended cheapest next experiment (no GPU commitment implied)
Skip the coarse-201 larger-dt ladder entirely. The FP32 cache control (this candidate) already targets the
actual blocker of the R6-anchored residual line: f16 cache quantization destroyed the residual signal
(prior ratios 0.164-3.058). If FP32 E_q = 0 confirms lossless storage, the next cheapest informative step is
a small preregistered fit-subset training probe on an fp32 cache slice to test whether the learned correction
beats the R54 baseline on development, BEFORE any 49.5 GiB full build.
