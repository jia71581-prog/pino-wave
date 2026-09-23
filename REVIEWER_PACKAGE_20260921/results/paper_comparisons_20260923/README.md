# TGRS paper comparison evidence — 2026-09-23

Four comparison groups (figures + tables + machine-readable numbers) built
CPU-only, writing only inside this directory. No GPU jobs launched; no
validation/test future wavefields read (all wavefield reads are train
records; the runtime JSONs cite pre-existing frozen artifacts).

Neural model under comparison: `scno_homogeneous_longrun_gap15_v1`,
update 5000, 9-record train-dev evaluation (frequency-gap dataset).
Historical anchor 29359 appears only in the runtime/accuracy context rows,
clearly labelled.

## Contents
- `group1_snapshots/` — 3x3 truth/prediction/difference panels,
  train_marmousi_00076 and train_layered_00299, early/middle/late physical
  times. PNG+PDF, NUMBERS.json, METHOD.md.
- `group2_receivers/` — near-surface receiver waveforms (z=40 m,
  x=400/800/1200/1600 m): truth vs neural + residual, and the dispersion
  overlay with the 51x51 coarse LWC solve. NUMBERS.json has per-receiver
  relL2 / xcorr lag / amplitude ratio.
- `group3_coarse_dispersion/` — the dataset's own LWC-84 solver re-run at
  51x51 (dx=40 m) and 101x101 (dx=20 m) on CPU, interpolated back to
  201x201, scored with the paper metric; dispersion figures (error growth,
  receiver spectrum, wavenumber spectrum). SOLVES.json + NUMBERS.json.
- `group4_tables/` — runtime table (frozen runtime_20260922 artifacts,
  record-matched ratios only, protocol footnotes) and DeepONet 40M baseline
  table (collapse reported verbatim, caliber mismatch stated).
- `scripts/` — the five generating scripts; each output cites its exact
  source paths.

## Key numbers
- Snapshot records, future relL2 (gap15 eval, verified to 1e-6 by
  recomputation): marmousi_00076 0.2785, layered_00299 0.1960.
- Coarse LWC vs neural (future relL2): marmousi 0.514 (51x51) / 0.120
  (101x101) vs 0.279 neural; layered 0.103 / 0.030 vs 0.196 neural.
- Dispersion signatures: coarse51 receiver lags +2.5..+7.5 ms and spectral
  centroid 12.97->10.66 Hz (marmousi); neural lag ~0 ms, error grows in time
  (0.12->0.41 across bands) — amplitude/late-coda deficit, not phase error.
- Runtime (RTX 4090 D, one record -> 401 frames): operator 3.97 s (tb=1) /
  2.54 s (tb=16) vs generation protocol 22.06 s, fine 20.07 s, native
  201x201 10.00 s. Record-matched ratios 5.06x/7.90x vs fine, 2.52x/3.93x vs
  native. No end-to-end 10x claim.
- DeepONet 40M: main run SIGTERM at step 4862/22814; fixed 12-record
  validation dense relL2 1.000012 (= trivial zero predictor level); 12-record
  probe control arm never < 1.0; anti-collapse arm inseparable from control.
  Failure attributed in the record to optimisation, not capacity
  (single-record fit reaches 0.025).

## Caliber findings (for the paper text)
1. 50x50 is not runnable: the solver requires odd node-centred grids;
   51x51/dx=40 m used instead (nearest admissible; documented).
2. Layered coarse solves beat the neural operator (vmin 2068 m/s keeps
   ~4.5 ppw at dx=40 m); marmousi (vmin 1500 m/s, ~3.3 ppw) shows the
   dispersion story. Reported as-is; do not cherry-pick marmousi only.
3. The cheapest classical config (native 201x201, 10.00 s) is more accurate
   than anchor 29359 on uniform/layered; the runtime table footnotes carry
   this so speed claims stay honest.
4. No same-caliber DeepONet number exists (run stopped early, never census-
   evaluated); the table pairs nearest calibers and labels each row.
5. The two figure records are train-dev (family medians of 3/family);
   validation/test remain untouched.
